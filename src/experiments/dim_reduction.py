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

from data_processing.data_reference import PropertyDataReference
from experiments.base import Experiment
from model.classification.base import AbstractClassifier
from model.reduction.weights import WeightsSelectorReducer
from model.classification.linear import LinearClassifier
from model.reduction.composite import CompositeReducer
from model.reduction.pca import TrainedPCAReducer
from utils.config import Configurations, Parameter
from view.plotter.scatter import ScatterPlotter, emb2plot
from utils.const import DEFAULT_TEMPLATES_SELECTED_NUMBER, DEFAULT_MAX_TOKENS_NUMBER

PROTECTED_PROPERTY: PropertyDataReference = ("religion", "protected", 3, 0)
STEREOTYPED_PROPERTY: PropertyDataReference = ("verb", "stereotyped", 1, 1)

configs = Configurations({
	Parameter.MAX_TOKENS_NUMBER: 'all',
	Parameter.TEMPLATES_SELECTED_NUMBER: 3,
	Parameter.CLASSIFIER_TYPE: 'linear',
	Parameter.CROSSING_STRATEGY: 'pppl',
	Parameter.POLARIZATION_STRATEGY: 'difference',
})


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
		
		# Getting embeddings
		protected_embedding_dataset, stereotyped_embedding_dataset = Experiment._get_embeddings(PROTECTED_PROPERTY, STEREOTYPED_PROPERTY, configs)

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
		folder: str = f'results/{PROTECTED_PROPERTY.name}-{STEREOTYPED_PROPERTY.name}'
		if not os.path.exists(folder):
			os.makedirs(folder)
		plot_data.to_csv(folder + f'/reduced_scatter_data_TM{DEFAULT_TEMPLATES_SELECTED_NUMBER}_TK{DEFAULT_MAX_TOKENS_NUMBER}_N{midstep}.csv', index=False)
		# ScatterPlotter(plot_data).show()
