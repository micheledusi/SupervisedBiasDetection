# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #


from enum import Enum
from model.binary_scoring.polarization.base import PolarizationScorer
from utils.config import Configurations, Parameter


class PolarizationStrategy(Enum):
	DIFFERENCE = 'difference'
	RATIO = 'ratio'


class PolarizationFactory:

	def __init__(self) -> None:
		raise NotImplementedError("This class cannot be instantiated.")
	
	@staticmethod
	def create(configs: Configurations) -> PolarizationScorer:
		"""
		This method creates a polarization scorer, given a configuration object.
		The configuration object must contain the following fields:
		- 'polarization_strategy': The type of operation that the polarization scorer must do.
		
		:param configs: The configuration object.
		:return: The polarization scorer.
		"""
		assert Parameter.POLARIZATION_STRATEGY in configs, f"The configuration object must contain the '{Parameter.POLARIZATION_STRATEGY}' field, in order to create a polarization scorer."
		strategy = configs[Parameter.POLARIZATION_STRATEGY]
		# Initialize the polarization strategy
		if strategy == PolarizationStrategy.DIFFERENCE or strategy == PolarizationStrategy.DIFFERENCE.value:
			operation = lambda x, y: x - y
		elif strategy == PolarizationStrategy.RATIO or strategy == PolarizationStrategy.RATIO.value:
			operation = lambda x, y: x / y
		return PolarizationScorer(binary_operator = operation)