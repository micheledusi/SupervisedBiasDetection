# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#							    #
#   Author:  Michele Dusi	    #
#   Date:	 2024			    #
# - - - - - - - - - - - - - - - #

# This module contains a class to perform a random dimensionality reduction.
# Each output feature is randomly selected from the input features.

import logging
import torch

from model.reduction.base import BaseDimensionalityReducer
from utils.config import Configurable, Parameter
from utils.const import DEVICE

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class RandomReducer(BaseDimensionalityReducer, Configurable):
	"""
	This class performs a dimensionality reduction by randomly selecting the output features.
	"""

	def __init__(self, configs):
		"""
		Initializer for the reducer class.

		:param configs: The configurations for the reducer.
		"""
		BaseDimensionalityReducer.__init__(self, requires_training=True)
		Configurable.__init__(self, configs, parameters=[
			Parameter.REDUCTION_DROPOUT_PERCENTAGE,
		], filtered=True)
		self._selected_features = None

	
	def _training_procedure(self, training_dataset: torch.Tensor, input_column: str, label_column: str, embeddings: torch.Tensor, labels: list) -> None:
		# Randomly select the features
		# We do not need to do use many parameters, byt we need to fix the features to select
		random_vector = torch.rand(size=(self.in_dim,))
		# It requires the "Reduction Dropout Percentage" parameter in the configurations.
		# Each feature is discarded with a probability equal to this parameter.
		self._selected_features = torch.where(random_vector > self.configs[Parameter.REDUCTION_DROPOUT_PERCENTAGE])[0]
		logger.info(f"Randomly selected features: {self._selected_features}")


	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		return torch.index_select(input=embeddings, dim=self._FEATURES_AXIS, index=self._selected_features)
	
	
	def get_transformation_matrix(self) -> torch.Tensor:
		matrix: torch.Tensor = torch.zeros(size=(self.in_dim, self.out_dim), dtype=torch.uint8).to(DEVICE)
		for i, feature in enumerate(self._selected_features):
			matrix[feature, i] = 1
		return matrix