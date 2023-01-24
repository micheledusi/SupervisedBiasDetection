# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#							    #
#   Author:  Michele Dusi	    #
#   Date:	2023			    #
# - - - - - - - - - - - - - - - #

# This module contains the base class for the dimensionality reducers.

import torch
from abc import ABC, abstractmethod


class BaseDimensionalityReducer(ABC):
	"""
	The Dimensionality Reducer takes a tensor with some features and reduces this number according to various criteria.
	"""

	_FEATURES_AXIS: int = -1

	def __init__(self, input_features: int, output_features: int):
		"""
		Initializer for the reducer class.

		:param input_features: The number of features of the input embeddings.
		:param output_features: The number of features of the output embeddings.
		"""
		self._in_dim = input_features
		self._out_dim = output_features

	@property
	def in_dim(self) -> int:
		"""
		The number of features of the input embeddings.

		:return: The number of features of the input embeddings.
		"""
		return self._in_dim
	
	@property
	def out_dim(self) -> int:
		"""
		The number of features of the output embeddings.

		:return: The number of features of the output embeddings.
		"""
		return self._out_dim

	@staticmethod
	def _count_features(embeddings: torch.Tensor) -> int:
		return embeddings.shape[BaseDimensionalityReducer._FEATURES_AXIS]

	@staticmethod
	def __prepare_input(embeddings: torch.Tensor) -> torch.Tensor:
		# NOTE: DEPRECATED
		# The embedding tensor are supposed to be already in the correct shape.
		#
		# if isinstance(embeddings, torch.Tensor):
		# 	embeddings = torch.squeeze(embeddings)
		return embeddings

	def __check_input(self, embeddings: torch.Tensor) -> None:
		assert self._count_features(embeddings) == self.in_dim, "The input embeddings have {} features, but the reducer expects {}.".format(self._count_features(embeddings), self.in_dim)

	def __check_output(self, embeddings: torch.Tensor) -> None:
		assert self._count_features(embeddings) == self.out_dim

	def reduce(self, embeddings: torch.Tensor) -> torch.Tensor:
		"""
		Applies the transformation of dimensions reduction, along the features' axis.
		The features' axis will change from length M to length N. The other dimensions will remain unchanged.
		
		By convention, the last dimension of the input tensor is the one containing the features.

		:param embeddings: The input tensor, of dimensions [ d1, d2, ..., dk, M ]
		:return: The output tensor, of dimensions [ d1, d2, ..., dk, N ]
		"""
		print(f"Reducing features from size = {self.in_dim:3d} to size = {self.out_dim:3d} with: ", type(self))
		embeddings = self.__prepare_input(embeddings)
		self.__check_input(embeddings)
		results = self._reduction_transformation(embeddings)
		self.__check_output(results)
		return results

	@abstractmethod
	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		"""
		Applies the transformation of dimensions reduction, along the features' axis.
		The features' axis will change from length M to length N. The other dimensions will remain unchanged.
		This method is not intended to be called directly, but by the reduce method, which performs some checks.
		"""
		raise NotImplementedError("This method must be implemented by the subclasses.")

	@abstractmethod
	def get_transformation_matrix(self) -> torch.Tensor:
		"""
		:return: The transformation matrix (if possible) representing the linear reduction of dimensionality.
		"""
		raise NotImplementedError("This method must be implemented by the subclasses.")


class MatrixReducer(BaseDimensionalityReducer):
	"""
	The reduction of this class is made by multiplying the embedding tensor by a MxN matrix, where:
	- M is the number of features of the input embeddings
	- N is the number of features of the output embeddings
	"""

	def __init__(self, matrix: torch.Tensor):
		assert len(matrix.shape) == 2
		input_features, output_features = matrix.shape
		super().__init__(input_features, output_features)
		self._matrix = matrix

	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		return torch.matmul(embeddings, self._matrix)

	def get_transformation_matrix(self) -> torch.Tensor:
		return self._matrix