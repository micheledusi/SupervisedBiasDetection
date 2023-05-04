# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	2023				#
# - - - - - - - - - - - - - - - #

from enum import Enum
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
	def __init__(self) -> None:
		raise NotImplementedError("This class cannot be instantiated.")

	@staticmethod
	def create(configs: Configurations) -> AbstractClassifier:
		"""
		This method creates a classifier, given a type and a set of arguments.
		The type must be one of the following:
		- 'linear': A linear classifier.
		- 'svm': A support vector machine classifier.

		:param type: The type of the classifier.
		:param kwargs: The arguments to be passed to the classifier constructor.
		:return: The classifier.
		"""
		clf_type = ClassifierType(configs.get(Parameter.CLASSIFIER_TYPE, DEFAULT_CLASSIFIER_TYPE))
		return clf_type.get_instance()