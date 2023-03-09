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
from model.classification.factory import ClassifierFactory
from model.reduction.weights import WeightsSelectorReducer
from model.reduction.composite import CompositeReducer
from model.reduction.pca import TrainedPCAReducer
from utils.config import Configurations, Parameter
from view.plotter.scatter import ScatterPlotter

PROTECTED_PROPERTY = PropertyDataReference("religion", "protected", 1, 0)
STEREOTYPED_PROPERTY = PropertyDataReference("verb", "stereotyped", 1, 1)

configs = Configurations({
	Parameter.MAX_TOKENS_NUMBER: 'all',
	Parameter.TEMPLATES_SELECTED_NUMBER: 'all',
	Parameter.CLASSIFIER_TYPE: 'svm',
})

MIDSTEP: int = 80


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
		prot_embs = protected_embedding_dataset['embedding']
		ster_embs = stereotyped_embedding_dataset['embedding']
		embs = torch.cat((prot_embs, ster_embs), dim=0)
		print("Embeddings obtained")
		print("Total number of embeddings:", len(embs))
		num_prot = len(prot_embs)
		num_ster = len(ster_embs)
		print(f"({num_prot} protected words + {num_ster} stereotyped words = {num_prot + num_ster} total words)")

		# 1. Reduction based on the weights of the classifier
		# 2. Reduction based on PCA
		classifier: AbstractClassifier = ClassifierFactory.create(configs)
		classifier.train(protected_embedding_dataset)
		reducer_1 = WeightsSelectorReducer.from_classifier(classifier, output_features=MIDSTEP)
		reduced_midstep_prot_embs = reducer_1.reduce(prot_embs)

		reducer_2: TrainedPCAReducer = TrainedPCAReducer(reduced_midstep_prot_embs, output_features=2)
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
		prot_prediction_dataset: Dataset = classifier.evaluate(protected_embedding_dataset)
		ster_prediction_dataset: Dataset = classifier.evaluate(stereotyped_embedding_dataset)
		predictions: torch.Tensor = torch.cat((prot_prediction_dataset['prediction'], ster_prediction_dataset['prediction']), dim=0)
		predicted_values: list[str] = [classifier.classes[p] for p in predictions]
		
		# Aggregating the results to show them
		results_ds = Dataset.from_dict({
			'word': protected_embedding_dataset['word'] + stereotyped_embedding_dataset['word'],
			'type': ['protected'] * num_prot + ['stereotyped'] * num_ster,
			'value': protected_embedding_dataset['value'] + stereotyped_embedding_dataset['value'],
			'x': reduced_embs[:, 0],
			'y': reduced_embs[:, 1],
			'prediction': predictions,
			'predicted_protected_value': predicted_values,
			})

		# If the directory does not exist, it will be created
		folder: str = f'results/{PROTECTED_PROPERTY.name}-{STEREOTYPED_PROPERTY.name}'
		if not os.path.exists(folder):
			os.makedirs(folder)
		configs_descriptor: str = configs.subget(Parameter.MAX_TOKENS_NUMBER, Parameter.TEMPLATES_SELECTED_NUMBER, Parameter.CLASSIFIER_TYPE).to_abbrstr()
		results_ds.to_csv(folder + f'/reduced_data_{configs_descriptor}_N{MIDSTEP}.csv', index=False)
		
		ScatterPlotter(results_ds, color_col='value').show()
