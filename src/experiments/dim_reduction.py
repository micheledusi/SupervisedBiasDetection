# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# EXPERIMENT: Dimensionality reduction over word embeddings obtained by BERT.
# DATE: 2023-01-20

import torch
from datasets import Dataset
from experiments.base import Experiment
from model.reduction.weights import WeightsSelectorReducer
from model.regression.linear_regressor import LinearRegressor
from model.reduction.composite import CompositeReducer
from model.reduction.pca import TrainedPCAReducer
from utility.cache_embedding import get_cached_embeddings
from data_processing.sentence_maker import PP_PATTERN, SP_PATTERN

class DimensionalityReductionExperiment(Experiment):

	def __init__(self) -> None:
		super().__init__("dimensionality reduction")

	def _execute(self, **kwargs) -> None:

		# Embeddings dataset
		protected_property = 'gender'
		protected_words_file = f'data/protected-p/{protected_property}/words-01.csv'
		protected_templates_file = f'data/protected-p/{protected_property}/templates-01.csv'
		protected_embedding_dataset = get_cached_embeddings(protected_property, PP_PATTERN, protected_words_file, protected_templates_file)
		protected_embedding = protected_embedding_dataset['embedding'].squeeze()	# [#words, #templates = 1, #tokens = 1, 768] -> [#words, 768]

		stereotyped_property = 'profession'
		stereotyped_words_file = f'data/stereotyped-p/{stereotyped_property}/words-01.csv'
		stereotyped_templates_file = f'data/stereotyped-p/{stereotyped_property}/templates-01.csv'
		stereotyped_embedding_dataset = get_cached_embeddings(stereotyped_property, SP_PATTERN, stereotyped_words_file, stereotyped_templates_file)
		stereotyped_embedding = stereotyped_embedding_dataset['embedding'].squeeze()	# [#words, #templates = 1, #tokens = 1, 768] -> [#words, 768]

		# Reducing the dimensionality of the embeddings
		midstep: int = 50

		# 1. Reduction based on the weights of the classifier
		# 2. Reduction based on PCA
		regressor: LinearRegressor = LinearRegressor()
		regressor.train(protected_embedding_dataset)

		reducer_1 = WeightsSelectorReducer.from_regressor(regressor, output_features=midstep)
		reduced_protected_embeddings = reducer_1.reduce(protected_embedding)

		reducer_2: TrainedPCAReducer = TrainedPCAReducer(reduced_protected_embeddings, output_features=2)

		reducer = CompositeReducer([
			reducer_1,
			reducer_2
		])

		# Reducing the embeddings
		results = reducer.reduce(stereotyped_embedding)
		print("Results: ", results)
		print("Results shape: ", results.shape)
		
