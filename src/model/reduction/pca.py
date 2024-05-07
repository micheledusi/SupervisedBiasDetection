# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module offers some classes to perform dimensionality reduction based on variance analysis.

import logging
import torch
from sklearn.decomposition import PCA
from datasets import Dataset

from model.reduction.base import BaseDimensionalityReducer
from utils.const import DEVICE


# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PCAReducer(BaseDimensionalityReducer):
	"""
	This class performs the reduction of the embeddings with Principal Component Analysis (PCA).
	PCA is a linear dimensionality reduction technique that uses Singular Value Decomposition (SVD) of the data to project it to a lower dimensional space.

	The effecs produced by this class depends only on the given input, and it has no memory of previous results.
	That is, different sets of embeddings will produce different results.

	For a more stable reduction able to store the transformation matrix, see the TrainedPCAReducer class.
	"""
	def __init__(self, output_features: int):
		super().__init__(output_features=output_features, requires_training=False)
		self._last_pca = None


	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		# Assuming that the embeddings are of shape: [#samples, #in_features]
		self._last_pca = PCA(n_components=self.out_dim)
		reduced_embeddings = self._last_pca.fit_transform(embeddings)
		return torch.Tensor(reduced_embeddings).to(DEVICE)


	def get_transformation_matrix(self) -> torch.Tensor:
		"""
		This method returns the transformation matrix used for the last reduction.
		
		:return: The transformation matrix.
		"""
		return torch.Tensor(self._last_pca.transform(torch.eye(self.in_dim))).to(DEVICE)


class JointPCAReducer(PCAReducer):
	"""
	This class performs the reduction of the embeddings with Principal Component Analysis (PCA).
	PCA is a linear dimensionality reduction technique that uses Singular Value Decomposition (SVD) of the data to project it to a lower dimensional space.

	The effecs produced by this class depends only on the given input, and it has no memory of previous results.
	That is, different sets of embeddings will produce different results.

	For a more stable reduction able to store the transformation matrix, see the TrainedPCAReducer class.
	"""
	def __init__(self, output_features: int):
		super().__init__(output_features=output_features)
		
	
	def _joint_reduction_transformation(self, prot_embeddings: torch.Tensor, ster_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Applies the transformation of dimensions reduction, along the features' axis, for both the protected and the stereotyped embeddings.
		The features' axis will change from length M to length N. The other dimensions will remain unchanged.
		This method is not intended to be called directly, but by the reduce method, which performs some checks.

		This is an override of the method in the superclass, to apply the reduction transformation 
		to both the protected and the stereotyped embeddings simultaneously. 
		The two embeddings are concatenated along the samples' axis, which is assumed to be the axis "-2".
		(The axis "-1" is the features' axis).
		"""
		logging.info(f"Reducing both the protected and stereotyped embeddings with PCA, to dimension = {self.out_dim}...")
		# Concatenate the two embeddings along the samples' axis
		embeddings = torch.cat((prot_embeddings, ster_embeddings), dim=-2)
		# Apply the PCA transformation
		reduced_embs = self._reduction_transformation(embeddings)
		# Split the embeddings back into the two parts
		tensors = torch.split(reduced_embs, [prot_embeddings.size(-2), ster_embeddings.size(-2)], dim=-2)
		for i, tensor in enumerate(tensors):
			logging.info(f"Reduced embeddings shape for the {['protected', 'stereotyped'][i]} embeddings: {tensor.shape}")
		return tensors


class TrainedPCAReducer(BaseDimensionalityReducer):
	"""
	This class performs the reduction of the embeddings with Principal Component Analysis (PCA).
	PCA is a linear dimensionality reduction technique that uses Singular Value Decomposition (SVD) of the data to project it to a lower dimensional space.

	Differently from the PCAReducer class, this class works in two steps:
	1. The transformation matrix is computed by reducing with PCA the training set of the initializer.
	2. The reduction is performed on the given embeddings, using the pre-computed matrix.
	"""
	def __init__(self, output_features: int):
		# Assuming that the train_embeddings are of shape: [#samples, #in_features]
		super().__init__(output_features=output_features, requires_training=True)
	

	def _training_procedure(self, training_dataset: Dataset, input_column: str, label_column: str, embeddings: torch.Tensor, labels: list) -> None:
		self._pca = PCA(n_components=self.out_dim)
		self._pca.fit(embeddings.cpu())


	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		reduced_embeddings = self._pca.transform(embeddings.cpu())
		return torch.Tensor(reduced_embeddings).to(DEVICE)