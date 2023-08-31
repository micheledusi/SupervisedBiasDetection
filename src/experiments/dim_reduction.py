# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# EXPERIMENT: Dimensionality reduction over word embeddings obtained by BERT.
# DATE: 2023-01-20

import os
from datasets import Dataset
import torch

from data_processing.data_reference import PropertyDataReference
from experiments.base import Experiment
from model.classification.base import AbstractClassifier
from model.classification.chi import ChiSquaredTest
from model.classification.factory import ClassifierFactory
from model.embedding.center import EmbeddingCenterer
from model.reduction.composite import CompositeReducer
from model.reduction.weights import WeightsSelectorReducer
from model.reduction.pca import TrainedPCAReducer
from model.reduction.tsne import TSNEReducer
from utils.config import Configurations, Parameter
from view.plotter.scatter import ScatterPlotter

configs = Configurations({
	Parameter.MAX_TOKENS_NUMBER: 'all',
	Parameter.TEMPLATES_SELECTED_NUMBER: 'all',
	Parameter.CLASSIFIER_TYPE: 'svm',
	Parameter.REDUCTION_TYPE: 'pca',
	Parameter.CENTER_EMBEDDINGS: False,
})

OUTPUT_NAME_REDUCED_EMBEDDINGS = "reduced_embeddings_classification"


class DimensionalityReductionExperiment(Experiment):
	"""
	This experiment aims at reducing the dimensionality of the embeddings obtained by BERT.
	The reduction is performed in two steps:
	1. Reduction based on the weights of the classifier
	2. Reduction based on PCA

	The weights of the classifier are obtained by training a linear regressor on the embeddings of the protected property.
	Then, the classifier is used to select the most relevant features of the embeddings.

	The PCA is performed on the embeddings of the protected property. The learned PCA is then applied to the embeddings of the stereotyped property.

	In this experiment, the embeddings are reduced to 2 dimensions.
	"""

	def __init__(self) -> None:
		super().__init__("dimensionality reduction", required_kwargs=['prot_prop', 'ster_prop', 'midstep'])

	def _execute(self, **kwargs) -> None:
		
		# Getting embeddings
		prot_dataset, ster_dataset = self._get_embeddings(configs)
		
		# Centering (optional)
		if configs[Parameter.CENTER_EMBEDDINGS]:
			centerer: EmbeddingCenterer = EmbeddingCenterer(configs)
			prot_dataset = centerer.center(prot_dataset)
			ster_dataset = centerer.center(ster_dataset)

		prot_embs = prot_dataset['embedding']
		ster_embs = ster_dataset['embedding']
		embs = torch.cat((prot_embs, ster_embs), dim=0)
		print("Embeddings obtained")
		print("Total number of embeddings:", len(embs))
		num_prot = len(prot_embs)
		num_ster = len(ster_embs)
		print(f"({num_prot} protected words + {num_ster} stereotyped words = {num_prot + num_ster} total words)")

		# 1. Reduction based on the weights of the classifier
		classifier: AbstractClassifier = ClassifierFactory.create(configs)
		classifier.train(prot_dataset)
		reducer_1 = WeightsSelectorReducer.from_classifier(classifier, output_features=self.midstep)
		reduced_midstep_prot_embs = reducer_1.reduce(prot_embs)

		# 2. Reduction based on PCA / t-SNE
		reducer_2 = None
		if configs[Parameter.REDUCTION_TYPE] == 'pca':
			reducer_2: TrainedPCAReducer = TrainedPCAReducer(reduced_midstep_prot_embs, output_features=2)
		elif configs[Parameter.REDUCTION_TYPE] == 'tsne':
			reducer_2: TSNEReducer = TSNEReducer(input_features=self.midstep, output_features=2)
		else:
			raise ValueError(f"Invalid reduction type: {configs[Parameter.REDUCTION_TYPE]}")
		
		# Combining the two reducers
		reducer = CompositeReducer([
			reducer_1,
			reducer_2
		])

		# Reducing the embeddings
		reduced_prot_embs: torch.Tensor = reducer.reduce(prot_embs)
		reduced_ster_embs: torch.Tensor = reducer.reduce(ster_embs)
		reduced_embs: torch.Tensor = torch.cat((reduced_prot_embs, reduced_ster_embs), dim=0)
		print("Reduction performed")
		print("Reduced protected embeddings shape:", reduced_prot_embs.shape)
		print("Reduced stereotyped embeddings shape:", reduced_ster_embs.shape)
		print("Reduced total embeddings shape:", reduced_embs.shape)

		# Predicting the values with the classifier
		prot_prediction_dataset: Dataset = classifier.evaluate(prot_dataset)
		ster_prediction_dataset: Dataset = classifier.evaluate(ster_dataset)
		predictions: torch.Tensor = torch.cat((prot_prediction_dataset['prediction'], ster_prediction_dataset['prediction']), dim=0)
		predicted_values: list[str] = [classifier.classes[p] for p in predictions]

		# Trying to predict the protected property with a new classifier, trained on the midstep-reduced embeddings
		midstep_classifier: AbstractClassifier = ClassifierFactory.create(configs)
		midstep_prot_dataset: Dataset = Dataset.from_dict({'embedding': reduced_midstep_prot_embs, 'value': prot_dataset['value']}).with_format('torch')
		midstep_ster_dataset: Dataset = Dataset.from_dict({'embedding': reducer_1.reduce(ster_embs), 'value': ster_dataset['value']}).with_format('torch')
		midstep_classifier.train(midstep_prot_dataset)
		midstep_prot_prediction_dataset: Dataset = midstep_classifier.evaluate(midstep_prot_dataset)
		midstep_ster_prediction_dataset: Dataset = midstep_classifier.evaluate(midstep_ster_dataset)
		midstep_predictions: torch.Tensor = torch.cat((midstep_prot_prediction_dataset['prediction'], midstep_ster_prediction_dataset['prediction']), dim=0)
		midstep_predicted_values: list[str] = [midstep_classifier.classes[p] for p in midstep_predictions]

		# Trying to predict the protected property with a new classifier, trained on the reduced embeddings
		reduced_classifier: AbstractClassifier = ClassifierFactory.create(configs)
		reduced_prot_dataset: Dataset = Dataset.from_dict({'embedding': reduced_prot_embs, 'value': prot_dataset['value']}).with_format('torch')
		reduced_ster_dataset: Dataset = Dataset.from_dict({'embedding': reduced_ster_embs, 'value': ster_dataset['value']}).with_format('torch')
		reduced_classifier.train(reduced_prot_dataset)
		reduced_prot_prediction_dataset: Dataset = reduced_classifier.evaluate(reduced_prot_dataset)
		reduced_ster_prediction_dataset: Dataset = reduced_classifier.evaluate(reduced_ster_dataset)
		reduced_predictions: torch.Tensor = torch.cat((reduced_prot_prediction_dataset['prediction'], reduced_ster_prediction_dataset['prediction']), dim=0)
		reduced_predicted_values: list[str] = [reduced_classifier.classes[p] for p in reduced_predictions]
		
		# Aggregating the results to show them
		results_ds = Dataset.from_dict({
			'word': prot_dataset['word'] + ster_dataset['word'],
			'type': ['protected'] * num_prot + ['stereotyped'] * num_ster,
			'value': prot_dataset['value'] + ster_dataset['value'],
			'x': reduced_embs[:, 0],
			'y': reduced_embs[:, 1],
			'original_prediction': predictions,
			'original_predicted_protected_value': predicted_values,
			'midstep_prediction': midstep_predictions,
			'midstep_predicted_protected_value': midstep_predicted_values,
			'reduced_prediction': reduced_predictions,
			'reduced_predicted_protected_value': reduced_predicted_values
			})

		# If the directory does not exist, it will be created
		folder: str = self._get_results_folder(configs, prot_dataset, ster_dataset)
		configs_descriptor: str = configs.to_abbrstr()
		filename: str = OUTPUT_NAME_REDUCED_EMBEDDINGS + f"_{configs_descriptor}_N{self.midstep}.csv"
		results_ds.to_csv(folder + '/' + filename, index=False)

		# Chi-squared test
		chi2 = ChiSquaredTest(verbose=True)
		filter_ster = lambda x: x['type'] == 'stereotyped'
		print("Original predictions, for N = 768:")
		original_chi_sq, original_p_value = chi2.test(results_ds.filter(filter_ster), 'value', 'original_predicted_protected_value')
		print(f"Midstep predictions, for N = {self.midstep}:")
		midstep_chi_sq, midstep_p_value = chi2.test(results_ds.filter(filter_ster), 'value', 'midstep_predicted_protected_value')
		print("Reduced predictions, for N = 2:")
		reduced_chi_sq, reduced_p_value = chi2.test(results_ds.filter(filter_ster), 'value', 'reduced_predicted_protected_value')

		print("Compared results:")
		print(f"Classification in the ORIGINAL embeddings:       p-value = {original_p_value:8.5e}     chi-squared = {original_chi_sq:10.8f}")
		print(f"Classification in the MIDSTEP  embeddings:       p-value = { midstep_p_value:8.5e}     chi-squared = { midstep_chi_sq:10.8f}")
		print(f"Classification in the REDUCED  embeddings:       p-value = { reduced_p_value:8.5e}     chi-squared = { reduced_chi_sq:10.8f}")
		print("(The lower the p-value, the higher the probability that the embeddings are stereotyped)\n")
		
		ScatterPlotter(results_ds, title=f"Reduced Embeddings (N = {self.midstep}, confidence = {100 - midstep_p_value*100:10.8f}%)", color_col='value').show()

		# Printing the result file path
		print("\nResults saved to CSV file:", f'{folder}/{filename}n')