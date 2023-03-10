# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module offers some classes to perform dimensionality reduction based on the t-SNE method.

import torch
from sklearn.manifold import TSNE

from model.reduction.base import BaseDimensionalityReducer
from utils.const import DEVICE


class TSNEReducer(BaseDimensionalityReducer):
	"""
	This class performs the reduction of the embeddings with t-SNE.
	t-SNE (t-distributed Stochastic Neighbor Embedding) is a nonlinear dimensionality reduction technique 
	that is particularly well suited for embedding high-dimensional data into a space of two or three dimensions, 
	which can then be visualized in a scatter plot.
	
	This class cannot be *trained*.
	Differently from the TrainedPCAReducer class (which uses PCA), t-SNE is not a transformation that can be expressed as a matrix
	or can be computed in a single step. Instead, it is a stochastic algorithm that requires multiple iterations to converge.
	Therefore, this class has no memory of previous results, and different sets of embeddings will produce different results.
	"""
	def __init__(self, input_features: int, output_features: int):
		super().__init__(input_features, output_features)
	
	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		self._tsne = TSNE(n_components=self.out_dim, perplexity=30, learning_rate=200, n_iter=1000)
		results = torch.Tensor(self._tsne.fit_transform(embeddings)).to(DEVICE)
		return results

	def get_transformation_matrix(self) -> torch.Tensor:
		raise NotImplementedError("This method is not implemented for this class, \
			    because the t-SNE transformation is not linear, thus it cannot be expressed as a matrix.\
			    Please, use the proper method of transformation (see the documentation of the class).")