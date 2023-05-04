# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# Analyzing the role of hyperparameter "n" in dimensionality reduction.
# Compared to the previous version of this experiment, this one considers
# multiple embeddings for each word. That means that we can compare the 
# embeddings obtained with different hyperparameters (e.g. number of templates, number of maximum tokens, etc.)
# and see how they perform in terms of bias detection.

import os
from datasets import Dataset
import torch

from tqdm import tqdm
from data_processing.data_reference import PropertyDataReference
from experiments.base import Experiment
from model.classification.base import AbstractClassifier
from model.classification.chi import ChiSquaredTest
from model.classification.factory import ClassifierFactory
from model.embedding.center import EmbeddingCenterer
from model.reduction.weights import WeightsSelectorReducer
from utils.config import ConfigurationsGrid, Parameter
from utils.const import DEVICE


# Configurations to process data
configurations = ConfigurationsGrid({
	Parameter.MAX_TOKENS_NUMBER: 'all',
	Parameter.TEMPLATES_SELECTED_NUMBER: 1,
	Parameter.CLASSIFIER_TYPE: 'svm',
	Parameter.CENTER_EMBEDDINGS: False,
})

MIDSTEPS: list[int] = list(range(2, 768+1))

PROTECTED_PROPERTY = PropertyDataReference("religion", "protected", 4, 1)
STEREOTYPED_PROPERTY = PropertyDataReference("quality", "stereotyped", 1, 1)
BIAS_GENERATION_ID = 1


class MidstepAnalysisChiSquared(Experiment):

	def __init__(self) -> None:
		super().__init__("midstep analysis with chi squared")

	def _execute(self, **kwargs) -> None:
		results: Dataset = Dataset.from_dict({"n": MIDSTEPS})

		for configs in configurations:
			# Showing the current configuration
			print("Current parameters configuration:\n", configs, '\n')
			
			# Getting the embeddings
			prot_dataset, ster_dataset = Experiment._get_embeddings(PROTECTED_PROPERTY, STEREOTYPED_PROPERTY, configs)
			
			# Centering (optional)
			if configs[Parameter.CENTER_EMBEDDINGS]:
				centerer: EmbeddingCenterer = EmbeddingCenterer(configs)
				prot_dataset = centerer.center(prot_dataset)
				ster_dataset = centerer.center(ster_dataset)
				
			prot_embs: torch.Tensor = prot_dataset['embedding'].to(DEVICE)
			ster_embs: torch.Tensor = ster_dataset['embedding'].to(DEVICE)

			# Creating and training the classifier
			classifier: AbstractClassifier = ClassifierFactory.create(configs)
			classifier.train(prot_dataset)

			# Creating the Chi-Squared tester
			chi2 = ChiSquaredTest(verbose=False)
			chi_squared_values = []
			p_values = []

			# For each column of polarization scores, we compute the chi-squared value
			for n in tqdm(MIDSTEPS):
				# Instantiate the reducer from 768 to N
				reducer_to_n = WeightsSelectorReducer.from_classifier(classifier, output_features=n)
				midstep_prot_embs = reducer_to_n.reduce(prot_embs).to(DEVICE)
				midstep_prot_dataset: Dataset = Dataset.from_dict({'embedding': midstep_prot_embs, 'value': prot_dataset['value']}).with_format('torch')
				midstep_ster_embs = reducer_to_n.reduce(ster_embs).to(DEVICE)
				midstep_ster_dataset: Dataset = Dataset.from_dict({'embedding': midstep_ster_embs, 'value': ster_dataset['value']}).with_format('torch')	
				# Now, we train another classifier on the reduced embeddings
				midstep_classifier = ClassifierFactory.create(configs)
				midstep_classifier.train(midstep_prot_dataset)
				midstep_ster_predictions = midstep_classifier.evaluate(midstep_ster_dataset)['prediction']
				midstep_predicted_values = [midstep_classifier.classes[pred] for pred in midstep_ster_predictions]
				midstep_ster_dataset = midstep_ster_dataset.add_column('midstep_predicted_value', midstep_predicted_values)
				# Now we can compute the chi-squared value
				chi_squared, p_value = chi2.test(midstep_ster_dataset, 'value', 'midstep_predicted_value')
				chi_squared_values.append(chi_squared)
				p_values.append(p_value)

			# Adding the results to the dataset
			results = results.add_column(f"chi_squared_{configs.subget_mutables().to_abbrstr()}", chi_squared_values)
			results = results.add_column(f"p_value_{configs.subget_mutables().to_abbrstr()}", p_values)

		# Save the results
		folder = f"results/{PROTECTED_PROPERTY.name}-{STEREOTYPED_PROPERTY.name}"
		if not os.path.exists(folder):
			os.makedirs(folder)
		# The filename will contain the IMMUTABLE parameters in the configuration, i.e.
		# the parameters that cannot change from one experiment to another.
		filename = f"aggregated_midstep_chi_squared_{configs.subget_immutables().to_abbrstr()}.csv"
		results.to_csv(f"{folder}/{filename}", index=False)



