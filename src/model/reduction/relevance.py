# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	 2024				#
# - - - - - - - - - - - - - - - #

import logging

import torch

from model.reduction.base import BaseDimensionalityReducer
from model.relevance.base import BaseRelevanceCalculator
from model.relevance.factory import RelevanceCalculatorFactory
from model.relevance.normalize import RelevanceNormalizer
from utils.config import Configurable, Parameter
from utils.const import DEVICE

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class RelevanceBasedReducer(BaseDimensionalityReducer, Configurable):
	"""
	This class performs a dimensionality reduction based on the relevance of the features.
	It can be specialized with different strategies by the configuration object.
	For example, it requires:
	- the strategy to compute the relevance of the features,
	- if the previous one is "from_classifier", the type of classifier to use,
	- the strategy to normalize the relevance values,
	- the strategy to select the most relevant features.

	This reducer requires also a training set to compute the relevance of the features.
	The training can be called with the 'train_ds' method, that requires the training set and the labels.
	"""

	def __init__(self, configs):
		"""
		Initializer for the reducer class.

		:param configs: The configurations for the reducer.
		"""
		BaseDimensionalityReducer.__init__(self, requires_training=True)
		Configurable.__init__(self, configs, parameters=[
			Parameter.RELEVANCE_COMPUTATION_STRATEGY,
			Parameter.RELEVANCE_CLASSIFIER_TYPE,
			Parameter.RELEVANCE_NORMALIZATION_STRATEGY,
			Parameter.RELEVANCE_FEATURES_SELECTION_STRATEGY,
		], filtered=True)
		self._selected_features = None


	def _training_procedure(self, training_dataset: torch.Tensor, input_column: str, label_column: str, embeddings: torch.Tensor, labels: list) -> None:
		# Extracting relevance from the training set
		factory: RelevanceCalculatorFactory = RelevanceCalculatorFactory(self.configs)
		relevance_calculator: BaseRelevanceCalculator = factory.create()
		relevance_scores: torch.Tensor = relevance_calculator.extract_relevance(training_dataset, input_column, label_column)
		# Normalizing the relevance scores
		normalized_relevance_scores: torch.Tensor = self._normalize(relevance_scores)
		# Selecting the most relevant features
		# The "selected_features" attribute will contain the indices of the selected features
		self._selected_features: torch.Tensor = self._select_features_indices(normalized_relevance_scores)

	
	def _normalize(self, relevance_scores: torch.Tensor) -> torch.Tensor:
		normalization_strategy = self.configs[Parameter.RELEVANCE_NORMALIZATION_STRATEGY]
		normalizer: RelevanceNormalizer = RelevanceNormalizer(relevance_scores)
		# Selecting the proper normalization strategy according to the configurations
		if normalization_strategy == "linear":
			return normalizer.linear()
		elif normalization_strategy == "linear_opposite":
			return normalizer.linear_opposite()
		elif normalization_strategy == "quadratic":
			return normalizer.squared()
		elif normalization_strategy == "quadratic_opposite":
			return normalizer.squared_opposite()
		elif normalization_strategy == "sigmoid":
			return normalizer.sigmoid()
		elif normalization_strategy == "sigmoid_opposite":
			return normalizer.sigmoid_opposite()
		elif normalization_strategy == "sigmoid_adaptive":
			return normalizer.sigmoid_adaptive()
		else:
			raise ValueError(f"Normalization strategy '{normalization_strategy}' not recognized.")


	def _select_features_indices(self, normalized_relevance_scores: torch.Tensor) -> torch.Tensor:
		features_selection_strategy = self.configs[Parameter.RELEVANCE_FEATURES_SELECTION_STRATEGY]
		# Selecting the most relevant features according to the configurations
		logger.info(f"Selecting the most relevant features with strategy '{features_selection_strategy}'.")
  
		if features_selection_strategy == "top_percentile":
			percentile = self.configs[Parameter.RELEVANCE_PERCENTILE_OR_THRESHOLD]
			num_out_features = round(self.in_dim * percentile)
			return torch.argsort(normalized_relevance_scores, descending=True)[:num_out_features]
		
		elif features_selection_strategy == "over_threshold":
			threshold = self.configs[Parameter.RELEVANCE_PERCENTILE_OR_THRESHOLD]
			# We return the indices of the features with relevance scores above the threshold
			return torch.nonzero(normalized_relevance_scores > threshold).squeeze()
		
		elif features_selection_strategy == "sampling":
			random_probs = torch.rand(size=normalized_relevance_scores.shape)
			# We return the indices of the features for which a random number is below the relevance score
			# This way, we sample the features with a probability proportional to their relevance
			return torch.nonzero(random_probs < normalized_relevance_scores).squeeze()

		else:
			raise ValueError(f"Features selection strategy '{features_selection_strategy}' not recognized.")


	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		return torch.index_select(input=embeddings.to(DEVICE), dim=self._FEATURES_AXIS, index=self._selected_features).to(DEVICE)