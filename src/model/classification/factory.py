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
from utils.config import Configurations, Parameter
from utils.const import DEFAULT_CLASSIFIER_TYPE


class ClassifierType(Enum):
	LINEAR = 'linear', LinearClassifier
	SVM = 'svm', SVMClassifier
	TREE = 'tree', TreeClassifier

	def __new__(cls, str_value: str, classifier_cls):
		obj = object.__new__(cls)
		obj._value_ = str_value
		obj._clf_cls_ = classifier_cls
		return obj

	def get_instance(self) -> AbstractClassifier:
		return self._clf_cls_()


class ClassifierFactory:
	"""
	This class is a factory for classifiers.
	"""

	PHASE_REDUCTION = 'reduction'
	PHASE_CROSS = 'cross'

	def __init__(self) -> None:
		raise NotImplementedError("This class cannot be instantiated.")

	@staticmethod
	def create(configs: Configurations, phase: str=PHASE_REDUCTION) -> AbstractClassifier:
		"""
		This method creates a classifier, given a type and a set of arguments.
		The type must be one of the following:
		- 'linear': A linear classifier.
		- 'svm': A support vector machine classifier.

		:param configs: The configurations to use.
		:param phase: The phase in which the classifier is being created. It can be either 'reduction' or 'cross'.
		:return: The classifier.
		"""
		if phase == ClassifierFactory.PHASE_REDUCTION:
			clf_type = ClassifierType(configs.get(Parameter.REDUCTION_CLASSIFIER_TYPE, DEFAULT_CLASSIFIER_TYPE))
			logging.info("Creating a reduction classifier of type '%s'...", clf_type.value)
		elif phase == ClassifierFactory.PHASE_CROSS:
			clf_type = ClassifierType(configs.get(Parameter.CROSS_CLASSIFIER_TYPE, DEFAULT_CLASSIFIER_TYPE))
			logging.info("Creating a cross-validation classifier of type '%s'...", clf_type.value)
		else:
			error_message = f"Invalid phase reference: '{phase}'"
			logging.error(error_message)
			raise ValueError(error_message)
		return clf_type.get_instance()