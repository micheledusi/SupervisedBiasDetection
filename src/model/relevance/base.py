# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#							    #
#   Author:  Michele Dusi	    #
#   Date:	 2024			    #
# - - - - - - - - - - - - - - - #

# This module contains the base class for the relevance computation.
# Computing the relevance means, in this project:
# - having a set of embeddings (i.e. float vectors)
# - having a set of labels associated with the embeddings, labeling each of them by classes
# - computing the relevance of each feature of the embeddings with respect to the classes

import logging
from typing import final
import torch
from abc import ABC, abstractmethod
from datasets import Dataset

from utils.config import Configurable, Configurations, Parameter
from utils.const import COL_CLASS, COL_EMBS

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class BaseRelevanceCalculator(ABC, Configurable):
	"""
	A generic relevance calculator is a class that computes the relevance of the features of a set of embeddings with respect to a set of classes.
	"""
	
	_UNINITIALIZED_DIMENSIONS: int = -1

	def __init__(self, configs: Configurations, num_features: int = _UNINITIALIZED_DIMENSIONS):
		"""
		Initializer for the reducer class.

        :param input_features: The number of features of the input embeddings. This value can be inserted if known and is desirable to verify it on execution. Otherwise it will be computed.
		"""
		ABC.__init__(self)
		Configurable.__init__(self, configs, parameters=[
			Parameter.RELEVANCE_COMPUTATION_STRATEGY, 
			Parameter.RELEVANCE_CLASSIFIER_TYPE,
			], filtered=False)
		logger.info(f"Initializing a relevance calculator with {num_features} features.")
		self.__dim = num_features
	

	@property
	def dim(self) -> int:
		"""
		The number of features for which the relevance is computed, only if it has been initialized.
		Otherwise, it throws an exception.

		:return: The number of features.
		"""
		if self.__dim == self._UNINITIALIZED_DIMENSIONS:
			logger.warning("Tried to access the number of features before initialization.")
			raise ValueError("The number of features has not been initialized.")
		return self.__dim


	def __count_features(self, embeddings: torch.Tensor) -> None:
		"""
		Counts the number of features of the input tensor and stores it in the `dim` attribute.
		
		:param embeddings: The input tensor, of dimensions [ d1, d2, ..., dk, M ]
		"""
		self.__dim = embeddings.shape[-1]


	@final
	def extract_relevance(self, dataset: Dataset, input_column: str=COL_EMBS, label_column: str=COL_CLASS) -> torch.Tensor:
		"""
		Computes the relevance of the features of the input embeddings.
		A feature is relevant if it is important for the classification of the embeddings, according to the classes.
		By convention, the last dimension of the input tensor is the one containing the features.

		:param dataset: The input dataset, containing the embeddings.
		:param input_column: The name of the column containing the embeddings.
		:param label_column: The name of the column containing the labels.
		:return: The output relevance vector, of dimensions [ M ]
		"""
		embeddings = dataset[input_column]
		if self.__dim == self._UNINITIALIZED_DIMENSIONS:
			self.__count_features(embeddings)
		logger.info(f"Extracting relevance from embeddings of length {self.dim}.")
		results = self._extraction(dataset, input_column, label_column)
		assert len(results.shape) == 1, f"Expected a relevance vector of dimension 1, but got {results.shape}."
		assert results.shape[0] == self.dim, f"Expected a relevance vector of length {self.dim}, but got {results.shape}."
		return results


	@abstractmethod
	def _extraction(self, dataset: Dataset, input_column: str=COL_EMBS, label_column: str=COL_CLASS) -> torch.Tensor:
		"""
		Applies the relevance extraction to the input tensor.
		This method must be implemented by the subclasses.

		:param dataset: The input dataset, containing the embeddings as a tensor of dimensions [ samples, M ]
		:param input_column: The name of the column containing the embeddings.
		:param label_column: The name of the column containing the labels.
		:return: The output relevance vector, of dimensions [ M ]
		"""
		raise NotImplementedError("This method must be implemented by the subclasses.")
