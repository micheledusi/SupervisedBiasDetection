# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2024				#
# - - - - - - - - - - - - - - - #

# This module contains the class "DisconnectionScorer", which is responsible for computing the 'disconnection' metrics between classes in a dataset of embeddings.


import torch
from datasets import Dataset

from utils.config import Configurable, Configurations, Parameter


DEFAULT_CLASS_COLUMN: str = "value"
DEFAULT_COORDS_COLUMN: str = "embedding"


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


class DisconnectionScorer(Configurable):
	"""
	This class is responsible for computing the 'disconnection' metrics between classes in a dataset of embeddings.
	"""

	# Separation metrics constants:
	ALPHA_SEPARATION_COEFFICIENT: float = 1.0
	BETA_COMPACTNESS_COEFFICIENT: float = 1.0
	DISCONNECTION_COEFFICIENT: float = ALPHA_SEPARATION_COEFFICIENT / BETA_COMPACTNESS_COEFFICIENT


	def __init__(self, configs: Configurations) -> None:
		super(DisconnectionScorer, self).__init__(configs, parameters=[
            Parameter.EMBEDDINGS_DISTANCE_STRATEGY,
			])
		
		if self.configs[Parameter.EMBEDDINGS_DISTANCE_STRATEGY] == 'cosine':
			self._distance_fn = cosine_distance
		elif self.configs[Parameter.EMBEDDINGS_DISTANCE_STRATEGY] == 'euclidean':
			self._distance_fn = euclidean_distance
		else:
			raise ValueError("Invalid distance strategy.")


	def compute_disconnection_score(self, embeddings_ds: Dataset, class_column: str=DEFAULT_CLASS_COLUMN, coords_col: str=DEFAULT_COORDS_COLUMN) -> float:
		"""
		Computes the "disconnection score" between classes in the dataset.
		The disconnection score is the ratio between the separation score and the compactness score, normalized by the ALPHA and BETA coefficients.
		
		:param embeddings_ds: the dataset of embeddings
		:param class_column: the name of the column containing the class labels
		:param coords_col: the name of the column containing the coordinates of the embeddings
		:return: the separation score
		"""
		separation_score: float = self.compute_separation_score(embeddings_ds, class_column, coords_col)
		compactness_score: float = self.compute_compactness_score(embeddings_ds, class_column, coords_col)
		return separation_score / compactness_score * self.DISCONNECTION_COEFFICIENT
	

	def compute_separation_score(self, embeddings: Dataset, class_col: str=DEFAULT_CLASS_COLUMN, coords_col: str=DEFAULT_COORDS_COLUMN) -> float:
		"""
		Computes the separation score of the given embeddings.

		:param embeddings: The embeddings to analyze, as a Dataset.
		:param class_col: The name of the column containing the class labels.
		:param coords_col: The name of the column containing the coordinates of the embeddings, as torch tensors.
		:return: The separation score of the given embeddings, as the average pairwise euclidean distance between inter-class embeddings.
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


	def compute_compactness_score(self, embeddings: Dataset, class_col: str=DEFAULT_CLASS_COLUMN, coords_col: str=DEFAULT_COORDS_COLUMN) -> float:
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
