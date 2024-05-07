# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#							    #
#   Author:  Michele Dusi	    #
#   Date:	2023			    #
# - - - - - - - - - - - - - - - #

# This module contains a class to perform a selection-based dimensionality reduction.

import torch

from model.reduction.base import BaseDimensionalityReducer
from utils.const import DEVICE


class SelectorReducer(BaseDimensionalityReducer):
	"""
	The reduction of this class is made by taking the features with specified indices.
	"""

	def __init__(self, input_features: int, indices: torch.Tensor):
		"""
		Initializer for the reducer class.

		:param input_features: The number of features of the input embeddings.
		:param indices: The selected indices for the output features.
		"""
		if indices.shape[-1] > 1:
			indices = indices.squeeze()
		BaseDimensionalityReducer.__init__(input_features=input_features, output_features=len(indices), requires_training=False)
		self._selected_features = indices.to(DEVICE)

	
	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		return torch.index_select(input=embeddings.to(DEVICE), dim=self._FEATURES_AXIS, index=self._selected_features).to(DEVICE)