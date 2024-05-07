# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	2023				#
# - - - - - - - - - - - - - - - #

from enum import Enum
import logging

from model.classification.base import AbstractClassifier

from model.classification.linear import LinearClassifier
from model.classification.svm import SVMClassifier
from model.classification.tree import TreeClassifier
from model.classification.random_forest import ForestClassifier
from utils.config import Configurable, Configurations, Parameter
from utils.const import DEFAULT_CLASSIFIER_TYPE


class ClassifierType(Enum):
	LINEAR = 'linear', LinearClassifier
	SVM = 'svm', SVMClassifier
	TREE = 'tree', TreeClassifier
	FOREST = 'randomforest', ForestClassifier

	def __new__(cls, str_value: str, classifier_cls):
		obj = object.__new__(cls)
		obj._value_ = str_value
		obj._clf_cls_ = classifier_cls
		return obj

	def get_instance(self) -> AbstractClassifier:
		return self._clf_cls_()


class ClassifierFactory(Configurable):
	"""
	This class is a factory for classifiers.
	"""

	PHASE_REDUCTION = 'reduction'
	PHASE_CROSS = 'cross'

	def __init__(self, configs: Configurations, phase: str=PHASE_REDUCTION) -> None:
		"""
		Initializer for the factory class.
		"""
		if phase == ClassifierFactory.PHASE_REDUCTION:
			Configurable.__init__(self, configs, parameters=[Parameter.RELEVANCE_CLASSIFIER_TYPE], filtered=False)
		elif phase == ClassifierFactory.PHASE_CROSS:
			Configurable.__init__(self, configs, parameters=[Parameter.CROSS_CLASSIFIER_TYPE], filtered=False)
		else:
			error_message = f"Invalid phase reference: '{phase}'"
			logging.error(error_message)
			raise ValueError(error_message)
		self._phase = phase

	def create(self) -> AbstractClassifier:
		"""
		This method creates a classifier, given a type and a set of arguments.
		The type must be one of the following:
		- 'linear': A linear classifier.
		- 'svm': A support vector machine classifier.
		- 'tree': A decision tree classifier.
		- 'randomforest': A random forest classifier.

		:param configs: The configurations to use.
		:param phase: The phase in which the classifier is being created. It can be either 'reduction' or 'cross'.
		:return: The classifier.
		"""
		if self._phase == ClassifierFactory.PHASE_REDUCTION:
			clf_type = ClassifierType(self.configs[Parameter.RELEVANCE_CLASSIFIER_TYPE])
			logging.info("Creating a reduction classifier of type '%s'...", clf_type.value)
		elif self._phase == ClassifierFactory.PHASE_CROSS:
			clf_type = ClassifierType(self.configs[Parameter.CROSS_CLASSIFIER_TYPE])
			logging.info("Creating a cross-validation classifier of type '%s'...", clf_type.value)
		return clf_type.get_instance()