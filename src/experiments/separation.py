# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# EXPERIMENT: Measuring separation between classes at the end of the dimensionality reduction process.
# DATE: 2023-07-12

import os
from datasets import Dataset, DatasetDict
import datasets
import torch
from tqdm import tqdm

from experiments.base import Experiment
from model.classification.base import AbstractClassifier
from model.classification.factory import ClassifierFactory
from model.embedding.center import EmbeddingCenterer
from model.reduction.pca import TrainedPCAReducer
from model.reduction.weights import WeightsSelectorReducer
from utils.config import Configurations, Parameter

MIDSTEPS: list[int] = list(range(2, 768+1))		# The list of midstep values for which the separation metrics will be computed

# Separation metrics constants:
ALPHA_SEPARATION_COEFFICIENT: float = 1.0
BETA_COMPACTNESS_COEFFICIENT: float = 1.0


class SeparationExperiment(Experiment):
	"""
	In this experiment, we perform a *dimensionality reduction* step over some property embeddings.
	The embeddings are obtained by BERT, and they are grouped into classes.
	Then, we measure the separation between the classes, by computing:
	- the average pairwise euclidean distance between inter-class embeddings (separation);
	- the average pairwise euclidean distance between intra-class embeddings (compactness).
	The final score is the ratio between the two.
	"""

	def __init__(self, configs: Configurations) -> None:
		super().__init__("separation measurement", required_kwargs=['prot_prop', 'midstep'], configs=configs)
		self._distance_fn = None
	
	def _execute(self, **kwargs) -> None:

		datasets.disable_caching()
		datasets.disable_progress_bar()

		embeddings_ds: Dataset = Experiment._get_property_embeddings(self.protected_property, self.configs)
		embeddings_ds = embeddings_ds.remove_columns(['descriptor','tokens'])
		embeddings_ds = embeddings_ds.add_column('labeled_value', embeddings_ds['value'])
		embeddings_ds = embeddings_ds.class_encode_column('labeled_value')

		if self.configs[Parameter.EMBEDDINGS_DISTANCE_STRATEGY] == 'cosine':
			# We need to center the embeddings
			if self.configs[Parameter.CENTER_EMBEDDINGS]:
				centerer: EmbeddingCenterer = EmbeddingCenterer(self.configs)
				embeddings_ds = centerer.center(embeddings_ds)
			else:
				raise ValueError("Cosine distance requires centered embeddings.")
			self._distance_fn = cosine_distance
		elif self.configs[Parameter.EMBEDDINGS_DISTANCE_STRATEGY] == 'euclidean':
			self._distance_fn = euclidean_distance
		else:
			raise ValueError("Invalid distance strategy.")

		### [1] 
		# We apply the dimensionality reduction step:
		reduced_embeddings_ds: Dataset = self._reduce(self.configs, self.midstep, embeddings_ds)

		# We also log the reduced dataset to a file
		# If the directory does not exist, it will be created
		folder: str = self._get_results_folder(self.configs, embeddings_ds)
		configs_descriptor: str = self.configs.to_abbrstr()
		# We split the coordinates in two columns
		printable_ds: Dataset = reduced_embeddings_ds.add_column('x', reduced_embeddings_ds['embedding'][:, 0].tolist())
		printable_ds = printable_ds.add_column('y', reduced_embeddings_ds['embedding'][:, 1].tolist())
		printable_ds = printable_ds.remove_columns('embedding')
		printable_ds.to_csv(folder + f'/reduced_embs_{configs_descriptor}_N{self.midstep}.csv', index=False)

		### [2] 
		# We compute the separation metrics:
		separation_score: float = self._compute_separation_score(reduced_embeddings_ds, "value", "embedding")
		compactness_score: float = self._compute_compactness_score(reduced_embeddings_ds, "value", "embedding")
		score: float = separation_score / compactness_score * ALPHA_SEPARATION_COEFFICIENT / BETA_COMPACTNESS_COEFFICIENT
		print(f"\nSeparation score: {separation_score}")
		print(f"Compactness score: {compactness_score}")
		print(f"\nFinal score: {score}")

		### [3]
		# In the second part of the experiment, we perform the same analysis over multiple "midstep" values.
		# We will use the same embeddings, but we will reduce them to different dimensions.
		scores_sep: list[float] = []
		scores_comp: list[float] = []
		scores: list[float] = []

		for n in tqdm(MIDSTEPS):
			reduced_embeddings_ds: Dataset = self._reduce(self.configs, n, embeddings_ds)
			separation_score: float = self._compute_separation_score(reduced_embeddings_ds, "value", "embedding")
			compactness_score: float = self._compute_compactness_score(reduced_embeddings_ds, "value", "embedding")

			scores_sep.append(separation_score)
			scores_comp.append(compactness_score)
			scores.append(separation_score / compactness_score * ALPHA_SEPARATION_COEFFICIENT / BETA_COMPACTNESS_COEFFICIENT)

		results: Dataset = Dataset.from_dict({
			"n": MIDSTEPS,
			"separation": scores_sep,
			"compactness": scores_comp,
			"separation_over_compactness": scores,
			})
		results.to_csv(folder + f'/separation_scores_{configs_descriptor}.csv', index=False)


	def _compute_separation_score(self, embeddings: Dataset, class_col: str, coords_col: str) -> float:
		"""
		Computes the separation score of the given embeddings.

		:param embeddings: The embeddings to analyze, as a Dataset.
		:param class_col: The name of the column containing the class labels.
		:param coords_col: The name of the column containing the coordinates of the embeddings, as torch tensors.
		:return: The separation score of the given embeddings, as the
		 		 average pairwise euclidean distance between inter-class embeddings.
		"""
		classes: list[str] = embeddings[class_col] # The list of classes
		coords: torch.Tensor = embeddings[coords_col] # The coordinates of the embeddings
		assert len(classes) == len(coords), "The number of classes and the number of embeddings must be the same."
		size: int = len(classes)

		distances: list[float] = []
		for i in range(size):
			for j in range(size):
				if i == j:
					continue
				# We need to check distance only between embeddings of different classes
				if classes[i] != classes[j]:
					dist: float = self._distance_fn(coords[i], coords[j]).item()
					distances.append(dist)
		return sum(distances) / len(distances)

	def _compute_compactness_score(self, embeddings: Dataset, class_col: str, coords_col: str) -> float:
		"""
		Computes the compactness score of the given embeddings.

		:param embeddings: The embeddings to analyze, as a Dataset.
		:param class_col: The name of the column containing the class labels.
		:param coords_col: The name of the column containing the coordinates of the embeddings, as torch tensors.
		:return: The compactness score of the given embeddings, as the
		 		 average pairwise euclidean distance between intra-class embeddings.
		"""
		classes: list[str] = embeddings[class_col]
		coords: torch.Tensor = embeddings[coords_col]
		assert len(classes) == len(coords), "The number of classes and the number of embeddings must be the same."
		size: int = len(classes)

		distances: list[float] = []
		for i in range(size):
			for j in range(size):
				if i == j:
					continue
				# We need to check distance only between embeddings of the same class
				if classes[i] == classes[j]:
					dist: float = self._distance_fn(coords[i], coords[j]).item()
					distances.append(dist)
		return sum(distances) / len(distances)

	def _reduce(self, configs: Configurations, midstep: int, embs_dataset: Dataset, embs_column: str = "embedding") -> Dataset:
		"""
		Performs a dimensionality reduction step over the given embeddings.
		The result will be the given dataset, without the original embeddings column, and with two new columns (x, y).
		"""
		
		# Reduction with our method:
		# 1. Train a classifier on the embeddings;
		# 2. Extract the weights of the classifier;
		# 3. Select the features of the embeddings according to the highest weights;
		# 4. Apply PCA to the shortened embeddings.
		classifier: AbstractClassifier = ClassifierFactory.create(configs)
		classifier.train(embs_dataset, input_column=embs_column)
		embs = embs_dataset[embs_column]
		reducer_1 = WeightsSelectorReducer.from_classifier(classifier, output_features=midstep)
		reduced_midstep_embs = reducer_1.reduce(embs)
		reducer_2 = TrainedPCAReducer(reduced_midstep_embs, output_features=2)
		reduced_embs = reducer_2.reduce(reduced_midstep_embs)

		result_ds: Dataset = embs_dataset.remove_columns(embs_column)
		result_ds = result_ds.add_column("embedding", reduced_embs.tolist())
		return result_ds.with_format("torch")
	

def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	"""
	Computes the euclidean distance between the two given tensors.
	"""
	return torch.dist(x, y, p=2)


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	"""
	Computes the cosine distance between the two given tensors.
	The result is 1 - cosine_similarity.
	"""
	return 1 - torch.cosine_similarity(x, y, dim=0)