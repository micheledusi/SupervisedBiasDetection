# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	 2024				#
# - - - - - - - - - - - - - - - #

from enum import Enum
import logging

from model.relevance.base import BaseRelevanceCalculator
from model.relevance.from_classifier import RelevanceFromClassification
from model.relevance.shap import RelevanceFromSHAP
from utils.config import Configurable, Configurations, Parameter

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RelevanceCalculationsStrategies(Enum):
	"""
	Enumeration of the possible strategies for relevance computation.
	"""
	FROM_CLASSIFIER = 'from_classifier', RelevanceFromClassification
	SHAP = 'shap', RelevanceFromSHAP

	def __new__(cls, str_value: str, calculator_cls):
		obj = object.__new__(cls)
		obj._value_ = str_value
		obj._clf_cls_ = calculator_cls	# Class of the relevance calculator
		return obj

	def get_instance(self, configs: Configurations) -> BaseRelevanceCalculator:
		return self._clf_cls_(configs)


class RelevanceCalculatorFactory(Configurable):
	"""
	This is a factory class for objects of type RelevanceCalculator.
	It means that it creates instances of RelevanceCalculator, depending on the configurations.
	"""	
	
	def __init__(self, configs: Configurations):
		"""
		Initializer for the factory class.
		"""
		Configurable.__init__(self, configs, parameters=[
			Parameter.RELEVANCE_COMPUTATION_STRATEGY, 
			Parameter.RELEVANCE_CLASSIFIER_TYPE,
			], filtered=False)


	def create(self):
		"""
		This method creates an instance of RelevanceCalculator, depending on the configurations.
		"""

		if self.configs[Parameter.RELEVANCE_COMPUTATION_STRATEGY] == RelevanceCalculationsStrategies.FROM_CLASSIFIER.value:
			from model.relevance.from_classifier import RelevanceFromClassification
			return RelevanceFromClassification(self.configs)
		
		elif self.configs[Parameter.RELEVANCE_COMPUTATION_STRATEGY] == RelevanceCalculationsStrategies.SHAP.value:
			from model.relevance.shap import RelevanceFromSHAP
			return RelevanceFromSHAP(self.configs)
		
		else:
			raise ValueError(f"Relevance computation strategy {self.configs[Parameter.RELEVANCE_COMPUTATION_STRATEGY]} not recognized.")

