# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	 2024				#
# - - - - - - - - - - - - - - - #

from enum import Enum
import logging

from model.reduction.base import BaseDimensionalityReducer
from utils.config import Configurable, Configurations, Parameter

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

VISUALIZATION_DIMENSION = 2
""" For the PCA reduction, the number of dimensions to reach. """


class ReductionStrategy(Enum):
	"""
	Enumeration of the possible strategies for reducing the embeddings.
	"""
	NONE = 'none'
	RANDOM = 'random'
	PCA = 'pca'
	TRAINED_PCA = 'trained_pca'
	TSNE = 'tsne'
	RELEVANCE_BASED = 'relevance_based'


class ReductionFactory(Configurable):
	"""
	This is a factory class for objects of type 'BaseDimensionalityReducer'.
	The correct reducer is chosen depending on the configurations.
	"""
	
	def __init__(self, configs: Configurations):
		"""
		Initializer for the factory class.
		"""
		logger.info("Initializing the ReductionFactory...")
		Configurable.__init__(self, configs, parameters=[
			Parameter.REDUCTION_STRATEGY,
			], filtered=False)
	

	def create(self) -> BaseDimensionalityReducer:
		"""
		This method creates an instance of 'BaseDimensionalityReducer', depending on the configurations.
		"""
		strategy = self.configs[Parameter.REDUCTION_STRATEGY]

		if strategy == ReductionStrategy.NONE.value:
			from model.reduction.base import IdentityReducer
			return IdentityReducer()
		
		elif strategy == ReductionStrategy.RANDOM.value:
			from model.reduction.random import RandomReducer
			return RandomReducer(self.configs)
		
		elif strategy == ReductionStrategy.PCA.value:
			from model.reduction.pca import PCAReducer
			return PCAReducer(VISUALIZATION_DIMENSION)
		
		elif strategy == ReductionStrategy.TRAINED_PCA.value:
			from model.reduction.pca import TrainedPCAReducer
			return TrainedPCAReducer(VISUALIZATION_DIMENSION)
		
		elif strategy == ReductionStrategy.TSNE.value:
			from model.reduction.tsne import TSNEReducer
			return TSNEReducer(VISUALIZATION_DIMENSION)
		
		elif strategy == ReductionStrategy.RELEVANCE_BASED.value:
			from model.reduction.relevance import RelevanceBasedReducer
			return RelevanceBasedReducer(self.configs)
		
		else:
			raise ValueError(f"Unknown reduction strategy: {strategy}")
		

	def create_multiple(self) -> dict[Configurations, BaseDimensionalityReducer]:
		"""
		This method creates multiple instances of 'BaseDimensionalityReducer', depending on the configurations.
		It adjusts automatically the number of reducers by combining the selected strategies and the parameters.
		More specifically, it iterates:
		- over the strategies, if they are multiple:
			- none
			- random
			- pca
			- trained_pca
			- tsne
			- relevance_based, for which it looks for
				- the relevance computation strategy; if it is "from_classifier", it also looks for
					- the classifier type
				- the normalization strategy
				- the features selection strategy
		
		:return: A list of instances of 'BaseDimensionalityReducer'.
		"""
		reducers: dict[Configurations, BaseDimensionalityReducer] = {}

		for config_strategy in self.configs.iterate_over([Parameter.REDUCTION_STRATEGY]):
			strategy = config_strategy[Parameter.REDUCTION_STRATEGY]

			# No-parameters strategies
			if strategy == ReductionStrategy.NONE.value \
					or strategy == ReductionStrategy.PCA.value \
					or strategy == ReductionStrategy.TRAINED_PCA.value \
					or strategy == ReductionStrategy.TSNE.value:
				new_reducer = ReductionFactory(config_strategy).create()
				filtered_config = config_strategy.subget_without(
					Parameter.REDUCTION_DROPOUT_PERCENTAGE,
					Parameter.RELEVANCE_COMPUTATION_STRATEGY,
					Parameter.RELEVANCE_CLASSIFIER_TYPE,
					Parameter.RELEVANCE_NORMALIZATION_STRATEGY,
					Parameter.RELEVANCE_FEATURES_SELECTION_STRATEGY,
					Parameter.RELEVANCE_PERCENTILE_OR_THRESHOLD,
					)
				reducers[filtered_config] = new_reducer

			# Random strategy, with "DROPOUT_PERCENTAGE" parameter
			elif strategy == ReductionStrategy.RANDOM.value:
				for config_strategy_dropout in config_strategy.iterate_over([Parameter.REDUCTION_DROPOUT_PERCENTAGE]):
					new_reducer = ReductionFactory(config_strategy_dropout).create()
					filtered_config = config_strategy_dropout.subget_without(
						Parameter.RELEVANCE_COMPUTATION_STRATEGY,
						Parameter.RELEVANCE_CLASSIFIER_TYPE,
						Parameter.RELEVANCE_NORMALIZATION_STRATEGY,
						Parameter.RELEVANCE_FEATURES_SELECTION_STRATEGY,
						Parameter.RELEVANCE_PERCENTILE_OR_THRESHOLD,
						)
					reducers[filtered_config] = new_reducer
			
			# Relevance-based strategy, with multiple parameters
			elif strategy == ReductionStrategy.RELEVANCE_BASED.value:
				for config_strategy_relevance in config_strategy.iterate_over([
						Parameter.RELEVANCE_COMPUTATION_STRATEGY,
						Parameter.RELEVANCE_NORMALIZATION_STRATEGY,
						Parameter.RELEVANCE_FEATURES_SELECTION_STRATEGY,
						Parameter.RELEVANCE_PERCENTILE_OR_THRESHOLD,
						]):
					if config_strategy_relevance[Parameter.RELEVANCE_COMPUTATION_STRATEGY] == 'from_classifier':
						for config_strategy_relevance_classifier in config_strategy_relevance.iterate_over([Parameter.RELEVANCE_CLASSIFIER_TYPE]):
							new_reducer = ReductionFactory(config_strategy_relevance_classifier).create()
							reducers[config_strategy_relevance_classifier] = new_reducer
					else:
						new_reducer = ReductionFactory(config_strategy_relevance).create()
						filtered_config = config_strategy_relevance.subget_without(
							Parameter.RELEVANCE_CLASSIFIER_TYPE,
							)
						reducers[filtered_config] = new_reducer

			else:
				raise ValueError(f"Unknown reduction strategy: {strategy}")
		
		if len(reducers) == 0:
			raise ValueError("No reduction strategy has been selected.")
		return reducers
	

	@staticmethod
	def get_strategy_str(config_strategy: Configurations) -> str:
		"""
		This method returns the string representation of the reduction strategy.

		:param config_strategy: The configurations for the reducer.
		:return: The string representation of the reduction strategy.
		"""

		strategy_base_str = config_strategy[Parameter.REDUCTION_STRATEGY]

		# No-parameters strategies
		if strategy_base_str in (
				ReductionStrategy.NONE.value, 
				ReductionStrategy.PCA.value, 
				ReductionStrategy.TRAINED_PCA.value, 
				ReductionStrategy.TSNE.value):
			return strategy_base_str

		# Random strategy, with "DROPOUT_PERCENTAGE" parameter
		elif strategy_base_str == ReductionStrategy.RANDOM.value:
			dropout = config_strategy[Parameter.REDUCTION_DROPOUT_PERCENTAGE]
			return f"{strategy_base_str} (dropout={dropout})"
		
		# Relevance-based strategy, with multiple parameters
		elif strategy_base_str == ReductionStrategy.RELEVANCE_BASED.value:
			relevance_computation = config_strategy[Parameter.RELEVANCE_COMPUTATION_STRATEGY]
			relevance_normalization = config_strategy[Parameter.RELEVANCE_NORMALIZATION_STRATEGY]
			relevance_features_selection = config_strategy[Parameter.RELEVANCE_FEATURES_SELECTION_STRATEGY]
			relevance_percentile_or_threshold = config_strategy[Parameter.RELEVANCE_PERCENTILE_OR_THRESHOLD]

			if relevance_computation == 'from_classifier':
				classifier = config_strategy[Parameter.RELEVANCE_CLASSIFIER_TYPE]
				base_str = f"{strategy_base_str} with scores from classifier={classifier}"
			elif relevance_computation == 'shap':
				base_str = f"{strategy_base_str} with scores from SHAP"
			base_str += f", normalization={relevance_normalization}, "

			if relevance_features_selection == 'top_percentile':
				base_str += f"selected top percentile={relevance_percentile_or_threshold}"
			elif relevance_features_selection == 'over_threshold':
				base_str += f"selected over threshold={relevance_percentile_or_threshold}"
			elif relevance_features_selection == 'sampling':
				base_str += f"selected by sampling"
			return base_str

		else:
			raise ValueError(f"Unknown reduction strategy: {strategy_base_str}")