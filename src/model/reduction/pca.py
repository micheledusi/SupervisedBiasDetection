# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module offers some classes to perform dimensionality reduction based on variance analysis.

import torch
from sklearn.decomposition import PCA

from model.reduction.base import BaseDimensionalityReducer
from utils.const import DEVICE


class PCAReducer(BaseDimensionalityReducer):
	"""
	This class performs the reduction of the embeddings with Principal Component Analysis (PCA).
	PCA is a linear dimensionality reduction technique that uses Singular Value Decomposition (SVD) of the data to project it to a lower dimensional space.

	The effecs produced by this class depends only on the given input, and it has no memory of previous results.
	That is, different sets of embeddings will produce different results.

	For a more stable reduction able to store the transformation matrix, see the TrainedPCAReducer class.
	"""
	def __init__(self, input_features: int, output_features: int):
		super().__init__(input_features, output_features)
		self._last_pca = None

	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		# Assuming that the embeddings are of shape: [#samples, #in_features]
		self._last_pca = PCA(n_components=self.out_dim)
		reduced_embeddings = self._last_pca.fit_transform(embeddings)
		return torch.Tensor(reduced_embeddings).to(DEVICE)

	def get_transformation_matrix(self) -> torch.Tensor:
		return torch.Tensor(self._last_pca.transform(torch.eye(self.in_dim))).to(DEVICE)


class TrainedPCAReducer(BaseDimensionalityReducer):
	"""
	This class performs the reduction of the embeddings with Principal Component Analysis (PCA).
	PCA is a linear dimensionality reduction technique that uses Singular Value Decomposition (SVD) of the data to project it to a lower dimensional space.

	Differently from the PCAReducer class, this class works in two steps:
	1. The transformation matrix is computed by reducing with PCA the training set of the initializer.
	2. The reduction is performed on the given embeddings, using the pre-computed matrix.
	"""
	def __init__(self, train_embeddings: torch.Tensor, output_features: int):
		# Assuming that the train_embeddings are of shape: [#samples, #in_features]
		super().__init__(train_embeddings.shape[1], output_features)
		self._pca = PCA(n_components=self.out_dim)
		self._pca.fit(train_embeddings)
	
	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		reduced_embeddings = self._pca.transform(embeddings)
		return torch.Tensor(reduced_embeddings).to(DEVICE)

	def get_transformation_matrix(self) -> torch.Tensor:
		return self._pca.transform(torch.eye(self.in_dim)).to(DEVICE)