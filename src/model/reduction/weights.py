# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module offers a class to perform dimensionality reduction based on the weights of the features.

import torch

from model.classification.base import AbstractClassifier
from model.reduction.selection import SelectorReducer


class WeightsSelectorReducer(SelectorReducer):
	"""
	This class performs a dimensionality reduction based on the weights of the features.
	The first K features with the highest weights are selected.

	NOTE: We assume that the weights relevance is an absolute value.
	For that, we consider the absolute value of the weights.
	"""

	def __init__(self, weights: torch.Tensor, output_features: int):
		"""
		Initializer for the reducer class.

		:param weights: The weights of the features.
		:param output_features: The number of features of the output embeddings, that is the number of selected features.
		"""
		assert len(weights.squeeze().size()) == 1, "The weights must be a 1D-vector."
		# Sort the weights in descending order
		indices = torch.argsort(weights.squeeze().abs(), descending=True)
		selected_indices = indices[:output_features]
		# Call the superclass constructor by passing the indices of the selected features, that is the highest K features.
		super().__init__(self._count_features(weights), selected_indices)
	
	@classmethod
	def from_weights(cls, weights: torch.Tensor, output_features: int) -> 'WeightsSelectorReducer':
		"""
		Factory method to create a new instance of the class.

		:param weights: The weights of the features.
		:param output_features: The number of features of the output embeddings, that is the number of selected features.
		:return: A new instance of the class.
		"""
		return cls(weights, output_features)
	
	@classmethod
	def from_classifier(cls, regressor: AbstractClassifier, output_features: int) -> 'WeightsSelectorReducer':
		"""
		Factory method to create a new instance of the class.

		:param regressor: The regressor that contains the weights.
		:param output_features: The number of features of the output embeddings, that is the number of selected features.
		:return: A new instance of the class.
		"""
		return cls.from_weights(regressor.features_relevance, output_features)