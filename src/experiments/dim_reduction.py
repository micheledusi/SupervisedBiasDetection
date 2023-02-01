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
from experiments.base import Experiment
from model.classification.abstract_classifier import AbstractClassifier
from model.reduction.weights import WeightsSelectorReducer
from model.classification.linear_classifier import LinearClassifier
from model.reduction.composite import CompositeReducer
from model.reduction.pca import TrainedPCAReducer
from view.plotter.scatter import ScatterPlotter, emb2plot
from utils.const import DEFAULT_TEMPLATES_SELECTED_NUMBER, DEFAULT_MAX_TOKENS_NUMBER


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
		super().__init__("dimensionality reduction")

	def _execute(self, **kwargs) -> None:
		protected_property = 'gender'
		stereotyped_property = 'profession'
		
		# Getting embeddings
		protected_embedding_dataset, stereotyped_embedding_dataset = Experiment._get_default_embeddings(protected_property, stereotyped_property)

		# Reducing the dimensionality of the embeddings
		midstep: int = 50

		# 1. Reduction based on the weights of the classifier
		# 2. Reduction based on PCA
		classifier: AbstractClassifier = LinearClassifier()
		classifier.train(protected_embedding_dataset)
		reducer_1 = WeightsSelectorReducer.from_classifier(classifier, output_features=midstep)
		reduced_protected_embeddings = reducer_1.reduce(protected_embedding_dataset['embedding'])

		reducer_2: TrainedPCAReducer = TrainedPCAReducer(reduced_protected_embeddings, output_features=2)

		reducer = CompositeReducer([
			reducer_1,
			reducer_2
		])

		# Reducing the embeddings
		results = reducer.reduce(stereotyped_embedding_dataset['embedding'])
		print("Results: ", results)
		print("Results shape: ", results.shape)
		
		# Showing the results
		results_ds = Dataset.from_dict({'word': stereotyped_embedding_dataset['word'], 'embedding': results, 'value': stereotyped_embedding_dataset['value']})
		plot_data: Dataset = emb2plot(results_ds)
		# If the directory does not exist, it will be created
		if not os.path.exists(f'results/{protected_property}-{stereotyped_property}'):
			os.makedirs(f'results/{protected_property}-{stereotyped_property}')
		plot_data.to_csv(f'results/{protected_property}-{stereotyped_property}/reduced_scatter_data_TM{DEFAULT_TEMPLATES_SELECTED_NUMBER}_TK{DEFAULT_MAX_TOKENS_NUMBER}_N{midstep}.csv', index=False)
		# ScatterPlotter(plot_data).show()
