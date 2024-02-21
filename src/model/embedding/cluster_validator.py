# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2024				#
# - - - - - - - - - - - - - - - #

# This module contains the class "DisconnectionScorer", which is responsible for computing the 'disconnection' metrics between classes in a dataset of embeddings.


from deprecated import deprecated
import torch
from datasets import Dataset

from utils.config import Configurable, Configurations, Parameter
from utils.const import COL_CLASS, COL_EMBS


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	"""
	Computes the cosine distance between the two given tensors.
	The result is 1 - cosine_similarity.
	"""
	return 1 - torch.cosine_similarity(x, y, dim=0)


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	"""
	Computes the euclidean distance between the two given tensors.
	"""
	return torch.dist(x, y, p=2)


def manhattan_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	"""
	Computes the manhattan distance between the two given tensors.
	"""
	return torch.dist(x, y, p=1)


class ClusteringScorer(Configurable):
	"""
	This class offers the method to compute the 'clustering validation metrics' between classes in a dataset of embeddings.
	"""


	def __init__(self, configs: Configurations) -> None:
		super(ClusteringScorer, self).__init__(configs, parameters=[
            Parameter.EMBEDDINGS_DISTANCE_STRATEGY,
			])
		
		if self.configs[Parameter.EMBEDDINGS_DISTANCE_STRATEGY] == 'cosine':
			self._distance_fn = cosine_distance
		elif self.configs[Parameter.EMBEDDINGS_DISTANCE_STRATEGY] == 'euclidean':
			self._distance_fn = euclidean_distance
		elif self.configs[Parameter.EMBEDDINGS_DISTANCE_STRATEGY] == 'manhattan':
			self._distance_fn = manhattan_distance
		else:
			raise ValueError("Invalid distance strategy.")


	def compute_clustering_score(self, embeddings_ds: Dataset, class_column: str=COL_CLASS, coords_column: str=COL_EMBS) -> float:
		"""
		Computes the "clustering score" between classes in the dataset.
		The clustering score is the ratio between the separation score and the compactness score, normalized by the ALPHA and BETA coefficients.
		
		:param embeddings_ds: the dataset of embeddings
		:param class_column: the name of the column containing the class labels
		:param coords_column: the name of the column containing the coordinates of the embeddings
		:return: the separation score
		"""
		classes: list[str] = embeddings_ds[class_column] # The list of classes
		coords: torch.Tensor = embeddings_ds[coords_column] # The coordinates of the embeddings
		size: int = len(classes)

		unique_classes: set[str] = set(classes)
		intra_class_distances: dict[str, list[float]] = {c: [] for c in unique_classes}
		inter_class_distances: dict[str, list[float]] = {c: [] for c in unique_classes}
		for i in range(size):
			current_ref_class: str = classes[i]
			for j in range(size):
				# In any case, we don't want to compare the same embeddings
				if i == j:
					continue
				# We compute the distance between the two embeddings
				dist: float = self._distance_fn(coords[i], coords[j]).item()
				if current_ref_class == classes[j]:
					# If the embedding [j] belongs to the same class of the reference embedding [i]
					intra_class_distances[current_ref_class].append(dist)
				else:
					# Otherwise, if the embedding [j] belongs to a different class than the reference embedding [i]
					inter_class_distances[current_ref_class].append(dist)

		dunn_indices: list[float] = []
		# For each cluster (=class)
		for c in unique_classes:
			# We compute the average distance between the cluster and the other clusters
			min_separation = min(inter_class_distances[c])
			# FIXME: UNUSED :: avg_separation = sum(inter_class_distances[c]) / len(inter_class_distances[c])
			# We compute the average distance within the cluster
			max_diameter = max(intra_class_distances[c])
			# FIXME: UNUSED :: avg_diameter = sum(intra_class_distances[c]) / len(intra_class_distances[c])
			# We compute the Dunn index for the cluster
			dunn_index = min_separation / max_diameter
			dunn_indices.append(dunn_index)
		# We compute the Dunn index for the entire dataset
		dunn_index = sum(dunn_indices) / len(dunn_indices)
		return dunn_index
	
		# TODO: Implement the silhouette score
	

	@deprecated
	def compute_separation_score(self, embeddings: Dataset, class_col: str=COL_CLASS, coords_col: str=COL_EMBS) -> float:
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


	@deprecated
	def compute_compactness_score(self, embeddings: Dataset, class_col: str=COL_CLASS, coords_col: str=COL_EMBS) -> float:
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
