# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	2023				#
# - - - - - - - - - - - - - - - #


from enum import Enum

from model.binary_scoring.crossing.base import CrossingScorer
from model.binary_scoring.crossing.mlm import MLMCrossScorer
from model.binary_scoring.crossing.pppl import PPPLCrossScorer
from utils.config import Configurations, Parameter


class CrossingStrategy(Enum):
	MLM = 'mlm', MLMCrossScorer
	PPPL = 'pppl', PPPLCrossScorer

	def __new__(cls, str_value: str, classifier_cls):
		obj = object.__new__(cls)
		obj._value_ = str_value
		obj._scorer_cls_ = classifier_cls
		return obj

	def get_instance(self, configs: Configurations) -> CrossingScorer:
		return self._scorer_cls_(configs)


class CrossingFactory:

	def __init__(self) -> None:
		raise NotImplementedError("This class cannot be instantiated.")

	@staticmethod
	def create(configs: Configurations) -> CrossingScorer:
		"""
		This method creates a cross-scorer, given a configuration object.
		The configuration object must contain the following fields:
		- 'cross_score_strategy': The type of cross-scorer to be created (MLM or PPPL).
		
		:param configs: The configuration object.
		:return: The cross-scorer.
		"""
		crossing_strategy = CrossingStrategy(configs[Parameter.CROSS_PROBABILITY_STRATEGY])
		return crossing_strategy.get_instance(configs)
	