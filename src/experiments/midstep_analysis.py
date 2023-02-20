# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# Analyzing the role of hyperparameter "n" in dimensionality reduction


import os
import torch
from torchmetrics import PearsonCorrCoef
from datasets import Dataset
from tqdm import tqdm
from experiments.base import Experiment
from model.classification.base import AbstractClassifier
from model.classification.factory import ClassifierFactory
from model.reduction.composite import CompositeReducer
from model.reduction.pca import TrainedPCAReducer
from model.reduction.weights import WeightsSelectorReducer
from utils.config import Configurations, Parameter
from view.plotter.scatter import ScatterPlotter, emb2plot
from utils.const import *


class MidstepAnalysisExperiment(Experiment):

	def __init__(self) -> None:
		super().__init__("midstep analysis")

	def _execute(self, **kwargs) -> None:
		protected_property = 'gender'
		stereotyped_property = 'profession'

		# Getting embeddings (as Dataset objects)
		protected_embedding_dataset, stereotyped_embedding_dataset = Experiment._get_default_embeddings(protected_property, stereotyped_property, rebuild=False)
		stereotyped_embedding_dataset = stereotyped_embedding_dataset.sort('word')

		# Getting MLM scores (as Dataset object)
		mlm_scores_ds = Experiment._get_default_mlm_scores(protected_property, stereotyped_property)
		mlm_scores_ds = mlm_scores_ds.rename_column('stereotyped_word', 'word')
		mlm_scores = mlm_scores_ds.sort('word')

		# Checking that the two datasets have the same words
		assert len(stereotyped_embedding_dataset) == len(mlm_scores), f"Expected the same number of words in the \"stereotyped embeddings dataset\" and in the \"MLM scores dataset\", but got {len(stereotyped_embedding_dataset)} and {len(mlm_scores)}."
		for w1, w2 in zip(stereotyped_embedding_dataset['word'], mlm_scores['word']):
			assert w1 == w2, f"Expected the same words in the \"stereotyped embeddings dataset\" and in the \"MLM scores dataset\", but got {w1} and {w2}."

		# TODO: this is going to be removed, when we will have more than one polarization axis
		num_polarizations = 0
		for column in mlm_scores.column_names:
			if column.startswith('polarization'):
				score_column = column
				num_polarizations += 1
		if num_polarizations > 1:
			raise NotImplementedError('The dataset "mlm_scores" contains more than one column representing a "polarization axis". This is not currently supported by this experiment.')
		# TODO END

		# Creating and training the classifier, with default parameters
		configs = Configurations()
		self._classifier: AbstractClassifier = ClassifierFactory.create(configs)
		self._classifier.train(protected_embedding_dataset)

		# For every value of midstep (called 'n'), run the experiment
		scores: dict = {'n': [], 'correlation': []}
		for n in tqdm(range(2, 768)):

			# First, we compute the 2D embeddings with the composite reducer, with the current value of midstep "n"
			# Note: if an exception occurs, we stop the experiment
			try:
				reduced_embeddings = self._reduce_with_midstep(protected_embedding_dataset, stereotyped_embedding_dataset, n)
			except RuntimeError as e:
				print(f"An exception occurred while reducing the embeddings with midstep {n}:\n{e}\nStopping the experiment.")
				break

			# Then, we compare the reduced embeddings with the mlm scores
			correlation = self._compute_correlation(reduced_embeddings, mlm_scores[score_column])
			scores['n'].append(n)
			scores['x-correlation'].append(correlation[0])
			scores['y-correlation'].append(correlation[1])
		
		scores: Dataset = Dataset.from_dict(scores)

		# Finally, we save the results
		folder = f"results/{protected_property}-{stereotyped_property}"
		if not os.path.exists(folder):
			os.makedirs(folder)
		filename = f"midstep_correlation_{configs.to_abbrstr(Parameter.CLASSIFIER_TYPE, Parameter.CROSSING_STRATEGY, Parameter.POLARIZATION_STRATEGY)}.csv"
		scores.to_csv(f"{folder}/{filename}", index=False)

	def _get_composite_reducer(self, prot_emb_ds: Dataset, midstep: int) -> CompositeReducer:
		"""
		Buils a composite reducer that first reduces the embeddings using the weights of the classifier and then
		reduces the result using PCA.
		The number of features in the first reduction is given by the parameter 'midstep'. At the end, the number of features is 2.

		:param prot_emb_ds: The dataset containing the protected embeddings
		:param midstep: The number of features to use in the first reduction
		"""
		# print("Number of embeddings: ", len(prot_emb_ds))
		# print("Embeddings dimension: ", prot_emb_ds['embedding'][0].shape)
		reducer_1 = WeightsSelectorReducer.from_classifier(self._classifier, output_features=midstep)
		reduced_protected_embeddings = reducer_1.reduce(prot_emb_ds['embedding'])
		# print("Reduced embeddings dimension: ", reduced_protected_embeddings[0].shape)
		reducer_2: TrainedPCAReducer = TrainedPCAReducer(reduced_protected_embeddings, output_features=2)
		# print("Reducer 2 input features: ", reducer_2.in_dim)
		reducer = CompositeReducer([reducer_1, reducer_2])
		return reducer
	
	def _reduce_with_midstep(self, prot_emb_ds: Dataset, ster_emb_ds: Dataset, midstep: int) -> torch.Tensor:
		reducer: CompositeReducer = self._get_composite_reducer(prot_emb_ds, midstep)
		return reducer.reduce(ster_emb_ds['embedding'])

	def _compute_correlation(self, reduced_embeddings: torch.Tensor, mlm_scores: torch.Tensor) -> torch.Tensor:
		"""
		Computes the correlation (a similarity measure) between the reduced embeddings and the MLM scores.
		The result is a tensor where each element is the correlation between the MLM scores and a coordinate of the reduced embeddings.

		:param reduced_embeddings: The reduced embeddings
		:param mlm_scores: The MLM scores
		"""
		assert reduced_embeddings.shape[0] == mlm_scores.shape[0], f"Expected the same number of embeddings and scores, but got {reduced_embeddings.shape[0]} and {mlm_scores.shape[0]}."
		
		coefs = []
		for polar_i in range(reduced_embeddings.shape[1]):
			# emb_coord = reduced_embeddings[:, polar_i]
			emb_coord = reduced_embeddings.moveaxis(0, 1)[polar_i]

			pearson = PearsonCorrCoef()
			corr = pearson(emb_coord, mlm_scores)
			coefs.append(corr.item())
		
		return torch.Tensor(coefs)




