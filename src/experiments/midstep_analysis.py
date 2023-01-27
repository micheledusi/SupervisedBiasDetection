# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# Analyzing the role of hyperparameter "n" in dimensionality reduction


import torch
from datasets import Dataset
from experiments.base import Experiment
from model.reduction.composite import CompositeReducer
from model.reduction.pca import TrainedPCAReducer
from model.reduction.weights import WeightsSelectorReducer
from model.regression.linear_regressor import LinearRegressor


class MidstepAnalysisExperiment(Experiment):

	def __init__(self) -> None:
		super().__init__("midstep analysis")

	def _execute(self, **kwargs) -> None:
		protected_property = 'gender'
		stereotyped_property = 'profession'

		# Getting embeddings
		protected_embedding_dataset, stereotyped_embedding_dataset = Experiment._get_default_embeddings(protected_property, stereotyped_property)

		# Getting MLM scores
		mlm_scores = Experiment._get_default_mlm_scores(protected_property, stereotyped_property)

		print("MLM scores:", mlm_scores)
		print("MLM scores data:", mlm_scores.data)
		exit()

		# For every value of n (called 'midstep'), run the experiment
		scores = []
		for midstep in range(2, 768):
			# First, we compute the 2D embeddings with the composite reducer
			reduced_embeddings = self._reduce_with_midstep(protected_embedding_dataset, stereotyped_embedding_dataset, midstep)
			# Then, we compute the "similarity" of the reduced embeddings, w.r.t. the original embeddings
			similarity = self._compute_similarity(stereotyped_embedding_dataset['embedding'], reduced_embeddings)
			scores.append(similarity)
		
		# Finally, we print the results
		print(scores)
	
	@staticmethod
	def _get_composite_reducer(prot_emb_ds: Dataset, midstep: int) -> CompositeReducer:
		"""
		Buils a composite reducer that first reduces the embeddings using the weights of the classifier and then
		reduces the result using PCA.
		The number of features in the first reduction is given by the parameter 'midstep'. At the end, the number of features is 2.

		:param prot_emb_ds: The dataset containing the protected embeddings
		:param midstep: The number of features to use in the first reduction
		"""
		regressor: LinearRegressor = LinearRegressor()
		regressor.train(prot_emb_ds)
		reducer_1 = WeightsSelectorReducer.from_regressor(regressor, output_features=midstep)
		reduced_protected_embeddings = reducer_1.reduce(prot_emb_ds['embedding'])
		reducer_2: TrainedPCAReducer = TrainedPCAReducer(reduced_protected_embeddings, output_features=2)
		reducer = CompositeReducer([reducer_1, reducer_2])
		return reducer
	
	@staticmethod
	def _reduce_with_midstep(prot_emb_ds: Dataset, ster_emb_ds: Dataset, midstep: int) -> torch.Tensor:
		reducer: CompositeReducer = MidstepAnalysisExperiment._get_composite_reducer(prot_emb_ds, midstep)
		return reducer.reduce(ster_emb_ds['embedding'])

	@staticmethod
	def _get_similarity(original_embeddings: torch.Tensor, reduced_embeddings: torch.Tensor) -> float:
		"""
		Computes the similarity between the original embeddings and the reduced embeddings.

		:param original_embeddings: The original embeddings
		:param reduced_embeddings: The reduced embeddings
		"""
		pass