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
from model.reduction.base import BaseDimensionalityReducer
from model.reduction.composite import CompositeReducer
from model.reduction.weights import WeightsSelectorReducer
from model.reduction.pca import TrainedPCAReducer
from model.reduction.tsne import TSNEReducer
from utils.config import Configurations, Parameter
from view.plotter.scatter import ScatterPlotter

PROTECTED_PROPERTY = PropertyDataReference("ethnicity", "protected", 1, 1)
STEREOTYPED_PROPERTY = PropertyDataReference("criminality", "stereotyped", 1, 1)

configs = Configurations({
	Parameter.MAX_TOKENS_NUMBER: 'all',
	Parameter.TEMPLATES_SELECTED_NUMBER: 'all',
	Parameter.CLASSIFIER_TYPE: 'svm',
	Parameter.REDUCTION_TYPE: 'pca',
	Parameter.CENTER_EMBEDDINGS: False,
})

MIDSTEP: int = 10


class DimensionalityReductionsComparisonExperiment(Experiment):
	"""
	This experiment aims at comparing different transformations of the embeddings obtained by BERT.
	Each transformation is a dimensionality reduction, at the end of which a classifier is trained 
	on the embeddings of the protected property and used to predict the values of the stereotyped property.

	Notice that the embeddings can be reduced to various numbers of dimensions, depending on the reduction method.
	"""

	def __init__(self) -> None:
		super().__init__("dimensionality reduction")

	def _execute(self, **kwargs) -> None:
		
		# Getting embeddings
		prot_dataset, ster_dataset = Experiment._get_embeddings(PROTECTED_PROPERTY, STEREOTYPED_PROPERTY, configs)
		
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

		# We start defining reducers to be compared
		reducers: list[BaseDimensionalityReducer] = []
		# Some of them will require a classifier trained on the protected property, thus we prepare it
		classifier: AbstractClassifier = ClassifierFactory.create(configs)
		classifier.train(prot_dataset)

		# 0. No reduction: 768 --> 768
		reducers.append(WeightsSelectorReducer.from_classifier(classifier, output_features=768))

		# 1. WeightsSelectorReducer: 768 --> MIDSTEP    	*** This is the one used in the paper ***
		reducers.append(WeightsSelectorReducer.from_classifier(classifier, output_features=MIDSTEP)) 

		# 2. WeightsSelectorReducer: 768 --> 2
		reducers.append(WeightsSelectorReducer.from_classifier(classifier, output_features=2))

		# 3. Reductor based on PCA/t-SNE: 768 --> min(len(prot_embs), MIDSTEP)
		out_dim = min(len(prot_embs), MIDSTEP)
		reducers.append(self._get_classic_reducer(prot_embs, out_dim=out_dim))

		# 4. Reductor based on PCA/t-SNE: 768 --> 2
		reducers.append(self._get_classic_reducer(prot_embs, out_dim=2))

		# 5. Composite reducer: 768 --> MIDSTEP --> 2
		reducer_1 = WeightsSelectorReducer.from_classifier(classifier, output_features=MIDSTEP)
		reducers.append(CompositeReducer([
			reducer_1,
			self._get_classic_reducer(reducer_1.reduce(prot_embs), out_dim=2)
		]))
		
		#####################

		# We prepare the results list
		results: list[tuple[BaseDimensionalityReducer, float, float]] = []

		# We iterate over the reducers
		for reducer in reducers:
			name = f"{type(reducer)}: {reducer.in_dim} --> {reducer.out_dim}"
			print(f"Reducing embeddings with {name}...")

			# Reducing the embeddings
			reduced_prot_embs: torch.Tensor = reducer.reduce(prot_embs)
			reduced_ster_embs: torch.Tensor = reducer.reduce(ster_embs)

			reduced_prot_embs_ds: Dataset = Dataset.from_dict({'embedding': reduced_prot_embs, 'value': prot_dataset['value']}).with_format('torch')
			reduced_ster_embs_ds: Dataset = Dataset.from_dict({'embedding': reduced_ster_embs, 'value': ster_dataset['value']}).with_format('torch')

			# Training a new classifier on the reduced embeddings
			reduced_classifier: AbstractClassifier = ClassifierFactory.create(configs)
			reduced_classifier.train(reduced_prot_embs_ds)

			# Predicting the values with the classifier
			prot_prediction_dataset: Dataset = reduced_classifier.evaluate(reduced_prot_embs_ds)
			prot_predicted_values: list[str] = [classifier.classes[p] for p in prot_prediction_dataset['prediction']]
			prot_prediction_dataset = prot_prediction_dataset.add_column('predicted_value', prot_predicted_values)
			
			ster_prediction_dataset: Dataset = reduced_classifier.evaluate(reduced_ster_embs_ds)
			ster_predicted_values: list[str] = [classifier.classes[p] for p in ster_prediction_dataset['prediction']]
			ster_prediction_dataset = ster_prediction_dataset.add_column('predicted_value', ster_predicted_values)

			# Computing the Chi-Squared test
			chi2 = ChiSquaredTest(verbose=True)
			chi_sq, p_value = chi2.test(ster_prediction_dataset, 'value', 'predicted_value')
			results.append((name, chi_sq, p_value))

		# Printing the results
		print("Results:")
		for result in results:
			print(f"{result[0]:70s}:    chi2 = {result[1]:6.3f},     p = {result[2]:6.3e}")
		print("")

		for result in results:
			print(f"{result[0]:70s}:    \cellChiP{'{'}{result[1]:.2f}{'}{'}{result[2]:.1e}{'}'}")
		print("")

	def _get_classic_reducer(self, embeddings: torch.Tensor, out_dim: int) -> BaseDimensionalityReducer:
		"""
		Returns a "classic" reducer, that is a reducer that does not use the weights of the classifier.
		It can be either a PCA reducer or a t-SNE reducer, based on the configuration.

		:param embeddings: The embeddings at the input of the reducer.
		:return: A "classic" reducer.
		"""
		if configs[Parameter.REDUCTION_TYPE] == 'pca':
			return TrainedPCAReducer(embeddings, output_features=out_dim)
		elif configs[Parameter.REDUCTION_TYPE] == 'tsne':
			in_dim = embeddings.shape[-1]
			return TSNEReducer(input_features=in_dim, output_features=out_dim)
		else:
			raise ValueError(f"Invalid reduction type: {configs[Parameter.REDUCTION_TYPE]}")