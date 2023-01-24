# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module offers some classes to perform dimensionality reduction based on variance analysis.

import torch

from model.reduction.base import BaseDimensionalityReducer, MatrixReducer


class PCAReducer(BaseDimensionalityReducer):
	"""
	This class performs the reduction of the embeddings with Principal Component Analysis (PCA).
	PCA is a linear dimensionality reduction technique that uses Singular Value Decomposition (SVD) of the data to project it to a lower dimensional space.

	The effecs produced by this class depends only on the given input, and it has no memory of previous results.
	That is, different sets of embeddings will produce different results.

	For a more stable reduction able to store the transformation matrix, see the TrainedPCAReducer class.
	"""

	_PCA_ITER_NUMBER: int = 3

	def __init__(self, input_features: int, output_features: int):
		super().__init__(input_features, output_features)
		self._transformation_matrix = None

	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		u, s, v = torch.pca_lowrank(embeddings, q=self.out_dim, niter=self._PCA_ITER_NUMBER)
		# Multiplying the original [..., #samples, #in_features] embedding matrix 
		# for the V transformation matrix of size: [#in_features, #out_features]
		self._transformation_matrix = v[:, :self.out_dim]
		return torch.matmul(embeddings, self._transformation_matrix)

	def get_transformation_matrix(self) -> torch.Tensor:
		return self._transformation_matrix


class TrainedPCAReducer(MatrixReducer):
	"""
	This class performs the reduction of the embeddings with Principal Component Analysis (PCA).
	PCA is a linear dimensionality reduction technique that uses Singular Value Decomposition (SVD) of the data to project it to a lower dimensional space.

	Differently from the PCAReducer class, this class works in two steps:
	1. The transformation matrix is computed by reducing with PCA the training set of the initializer.
	2. The reduction is performed on the given embeddings, using the pre-computed matrix.
	"""

	def __init__(self, train_embeddings: torch.Tensor, output_features: int):
		u, s, v = torch.pca_lowrank(train_embeddings, q=output_features, niter=PCAReducer._PCA_ITER_NUMBER)
		super().__init__(v[:, :output_features])