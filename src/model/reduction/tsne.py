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

	def __init__(self, output_features: int):
		super().__init__(output_features=output_features, requires_training=False)


	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		self._tsne = TSNE(n_components=self.out_dim, perplexity=30, learning_rate=200, n_iter=1000)
		results = torch.Tensor(self._tsne.fit_transform(embeddings)).to(DEVICE)
		return results
	

	def _joint_reduction_transformation(self, prot_embeddings: torch.Tensor, ster_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Applies the transformation of dimensions reduction, along the features' axis, for both the protected and the stereotyped embeddings.
		The features' axis will change from length M to length N. The other dimensions will remain unchanged.
		This method is not intended to be called directly, but by the reduce method, which performs some checks.

		This is an override of the method in the superclass, to apply the t-SNE transformation 
		to both the protected and the stereotyped embeddings simultaneously. 
		The two embeddings are concatenated along the samples' axis, which is assumed to be the axis "-2".
		(The axis "-1" is the features' axis).
		"""
		# Concatenate the two embeddings along the samples' axis
		embeddings = torch.cat((prot_embeddings, ster_embeddings), dim=-2)
		# Apply the t-SNE transformation
		reduced_embs = self._reduction_transformation(embeddings)
		# Split the embeddings back into the two parts
		return torch.split(reduced_embs, [prot_embeddings.size(-2), ster_embeddings.size(-2)], dim=-2)


	def get_transformation_matrix(self) -> torch.Tensor:
		raise NotImplementedError("This method is not implemented for this class, \
			    because the t-SNE transformation is not linear, thus it cannot be expressed as a matrix.\
			    Please, use the proper method of transformation (see the documentation of the class).")