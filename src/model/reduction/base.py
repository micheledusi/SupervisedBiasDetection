# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#							    #
#   Author:  Michele Dusi	    #
#   Date:	2023			    #
# - - - - - - - - - - - - - - - #

# This module contains the base class for the dimensionality reducers.

import logging
from typing import final
import torch
from abc import ABC, abstractmethod
from datasets import Dataset

from utils.const import COL_CLASS, COL_EMBS

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class BaseDimensionalityReducer(ABC):
	"""
	The Dimensionality Reducer takes a tensor with some features and reduces this number according to various criteria.
	"""

	_UNINITIALIZED_DIMENSION: int = -1
	_FEATURES_AXIS: int = -1

	def __init__(self, input_features: int = _UNINITIALIZED_DIMENSION, output_features: int = _UNINITIALIZED_DIMENSION, requires_training: bool = False):
		"""
		Initializer for the reducer class.

		:param input_features: The number of features of the input embeddings.
		:param output_features: The number of features of the output embeddings.
		:param requires_training: True if the reducer requires training, False otherwise.
		"""
		logger.info("Initializing the BaseDimensionalityReducer...")
		self._in_dim = input_features
		self._out_dim = output_features
		self._requires_training = requires_training


	@property
	def in_dim(self) -> int:
		"""
		The number of features of the input embeddings.

		:return: The number of features of the input embeddings.
		"""
		if self._in_dim == BaseDimensionalityReducer._UNINITIALIZED_DIMENSION:
			raise ValueError("The input dimension has not been initialized.")
		return self._in_dim
	

	@property
	def out_dim(self) -> int:
		"""
		The number of features of the output embeddings.

		:return: The number of features of the output embeddings.
		"""
		if self._out_dim == BaseDimensionalityReducer._UNINITIALIZED_DIMENSION:
			raise ValueError("The output dimension has not been initialized.")
		return self._out_dim


	@staticmethod
	def _count_features(embeddings: torch.Tensor) -> int:
		return embeddings.shape[BaseDimensionalityReducer._FEATURES_AXIS]
	

	def __check_input(self, embeddings: torch.Tensor) -> None:
		embs_dim = BaseDimensionalityReducer._count_features(embeddings)
		if self._in_dim == BaseDimensionalityReducer._UNINITIALIZED_DIMENSION:
			# If it is the first time we see the input embeddings, we store their dimension
			self._in_dim = embs_dim
		else:
			# We check that the input embeddings have the expected number of features
			assert embs_dim == self.in_dim, f"The input embeddings have {embs_dim} features, but the reducer expects {self.in_dim}."
		logger.debug(f"Device of the input embeddings: {embeddings.device}")


	def __check_output(self, embeddings: torch.Tensor) -> None:
		embs_dim = BaseDimensionalityReducer._count_features(embeddings)
		if self._out_dim == BaseDimensionalityReducer._UNINITIALIZED_DIMENSION:
			# If it is the first time we see the output embeddings, we store their dimension
			self._out_dim = embs_dim
		else:
			# We check that the output embeddings have the expected number of features
			assert embs_dim == self.out_dim, f"The output embeddings have {embs_dim} features, but the reducer expects {self.out_dim}."
		logger.debug(f"Device of the output embeddings: {embeddings.device}")
	

	@property
	def requires_training(self) -> bool:
		"""
		:return: True if the reducer requires training, False otherwise.
		"""
		return self._requires_training

	
	def train_ds(self, training_dataset: Dataset, input_column: str=COL_EMBS, label_column: str=COL_CLASS) -> None:
		"""
		Trains the reducer on the given dataset.
		This method is not intended to be called directly, but by the subclasses that require training.

		:param dataset: The dataset containing the embeddings.
		:param input_column: The name of the column containing the embeddings.
		"""
		if not self.requires_training:
			raise TypeError(f"This reducer of type <{type(self).__name__}> does not require training.")
		logger.info(f"Training the reducer of type: {type(self).__name__}")
		embeddings = training_dataset[input_column]
		labels = training_dataset[label_column]
		self.__check_input(embeddings)
		self._training_procedure(training_dataset, input_column, label_column, embeddings, labels)

	
	def _training_procedure(self, training_dataset: Dataset, input_column: str, label_column: str, embeddings: torch.Tensor, labels: list) -> None:
		"""
		Trains the reducer on the given dataset.
		This method is not intended to be called directly, but by the train_ds method, which performs some checks.

		:param dataset: The dataset containing the embeddings.
		:param input_column: The name of the column containing the embeddings.
		:param label_column: The name of the column containing the labels.
		:param embeddings: The embeddings contained in the dataset.
		:param labels: The labels contained in the dataset.
		"""
		raise NotImplementedError("This method must be implemented by the subclasses.")
	

	@abstractmethod
	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		"""
		Applies the transformation of dimensions reduction, along the features' axis.
		The features' axis will change from length M to length N. The other dimensions will remain unchanged.
		This method is not intended to be called directly, but by the reduce method, which performs some checks.
		"""
		raise NotImplementedError("This method must be implemented by the subclasses.")


	def _joint_reduction_transformation(self, prot_embeddings: torch.Tensor, ster_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Applies the transformation of dimensions reduction, along the features' axis, for both the protected and the stereotyped embeddings.
		The features' axis will change from length M to length N. The other dimensions will remain unchanged.
		This method is not intended to be called directly, but by the reduce method, which performs some checks.

		By default, this method applies the reduction to the protected embeddings and to the stereotyped embeddings separately.
		If a different behavior is needed, the subclasses should override this method, not the reduce method.
		"""
		return self._reduction_transformation(prot_embeddings), self._reduction_transformation(ster_embeddings)


	def get_transformation_matrix(self) -> torch.Tensor:
		"""
		Returns the transformation matrix (if possible) representing the linear reduction of dimensionality.
		By default, this method returns the result of the transformation of the identity matrix.
		Hopefully, the method can be overridden by the subclasses to return a different transformation matrix.

		:return: The transformation matrix (if possible) representing the linear reduction of dimensionality.
		"""
		return self._reduction_transformation(torch.eye(self.in_dim))


	@final
	def reduce_embs(self, embeddings: torch.Tensor) -> torch.Tensor:
		"""
		Applies the transformation of dimensions reduction, along the features' axis.
		The features' axis will change from length M to length N. The other dimensions will remain unchanged.
		
		By convention, the last dimension of the input tensor is the one containing the features.

		:param embeddings: The input tensor, of dimensions [ d1, d2, ..., dk, M ]
		:return: The output tensor, of dimensions [ d1, d2, ..., dk, N ]
		"""
		logger.info(f"Reducing features from size = {self.in_dim:3d} to size = {self.out_dim:3d} with: {type(self).__name__}")
		self.__check_input(embeddings)
		results = self._reduction_transformation(embeddings)
		self.__check_output(results)
		return results
	

	@final
	def reduce_both_embs(self, prot_embeddings: torch.Tensor, ster_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Applies the transformation of dimensions reduction, along the features' axis, to both the protected and the stereotyped embeddings.
		The features' axis will change from length M to length N. The other dimensions will remain unchanged.
		By convention, the last dimension of the input tensor is the one containing the features.

		By default, this method applies the reduction to the protected embeddings and to the stereotyped embeddings separately.
		If a different behavior is needed, the subclasses should override the method `_joint_reduction_transformation`, not this one.

		:param prot_embeddings: The 1st input tensor, of dimensions [ d11, d12, ..., d1k, M ]
		:param ster_embeddings: The 2nd input tensor, of dimensions [ d21, d22, ..., d2k, M ]
		:return: The tuple of output tensors: ( [ d11, d12, ..., d1k, N ], [ d21, d22, ..., d2k, N ] )
		"""
		self.__check_input(prot_embeddings)
		self.__check_input(ster_embeddings)
		reduced_prot_embeddings, reduced_ster_embeddings = self._joint_reduction_transformation(prot_embeddings, ster_embeddings)
		self.__check_output(reduced_prot_embeddings)
		self.__check_output(reduced_ster_embeddings)
		logger.info(f"Features reduced from size = {self.in_dim:3d} to size = {self.out_dim:3d} with: {type(self).__name__}")
		return reduced_prot_embeddings, reduced_ster_embeddings
	

	@final
	def reduce_ds(self, dataset: Dataset, input_column: str=COL_EMBS, output_column: str=COL_EMBS) -> Dataset:
		"""
		Applies the reduction to the embeddings contained in the given dataset.
		The input column will be removed, and a new column with the reduced embeddings will be added.

		:param dataset: The dataset containing the embeddings.
		:param input_column: The name of the column containing the embeddings.
		:param output_column: The name of the column where the reduced embeddings will be stored. If the column already exists, it will be overwritten.
		:return: The dataset with the reduced embeddings.
		"""
		embeddings = dataset[input_column]
		reduced_embeddings = self.reduce_embs(embeddings)
		return dataset.remove_columns(input_column).add_column(output_column, reduced_embeddings.tolist())
	
	
	@final
	def reduce_both_ds(self, prot_dataset: Dataset, ster_dataset: Dataset, input_column: str=COL_EMBS, output_column: str=COL_EMBS) -> tuple[Dataset, Dataset]:
		"""
		Applies the reduction to two datasets, one for the protected embeddings and one for the stereotyped embeddings.
		For both datasets, the input column will be removed, and a new column with the reduced embeddings will be added.
		The method assumes that the two datasets have the same name of the input column.

		Furthermore, this method uses the `reduce_both` method, which applies the reduction to the embeddings separately.
		If a different behavior is needed, the subclasses should override the method `reduce_both`, not this one.

		:param prot_dataset: The dataset containing the protected embeddings.
		:param ster_dataset: The dataset containing the stereotyped embeddings.
		:param input_column: The name of the column containing the embeddings.
		:param output_column: The name of the column where the reduced embeddings will be stored. If the column already exists, it will be overwritten.
		:return: The dataset with the reduced embeddings.
		"""
		prot_embs = prot_dataset[input_column]
		ster_embs = ster_dataset[input_column]
		reduced_prot_embs, reduced_ster_embs = self.reduce_both_embs(prot_embs, ster_embs)
		prot_dataset = prot_dataset.remove_columns(input_column).add_column(output_column, reduced_prot_embs.tolist())
		ster_dataset = ster_dataset.remove_columns(input_column).add_column(output_column, reduced_ster_embs.tolist())
		return prot_dataset, ster_dataset


class MatrixReducer(BaseDimensionalityReducer):
	"""
	The reduction of this class is made by multiplying the embedding tensor by a MxN matrix, where:
	- M is the number of features of the input embeddings
	- N is the number of features of the output embeddings
	"""

	def __init__(self, matrix: torch.Tensor):
		assert len(matrix.shape) == 2
		input_features, output_features = matrix.shape
		super().__init__(input_features, output_features, requires_training=False)
		self._matrix = matrix


	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		return torch.matmul(embeddings, self._matrix)


	def get_transformation_matrix(self) -> torch.Tensor:
		return self._matrix


class IdentityReducer(BaseDimensionalityReducer):
	"""
	The reduction of this class is the identity function, i.e., the input tensor is returned unchanged.
	"""

	def __init__(self):
		super().__init__(requires_training=False)


	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		return embeddings