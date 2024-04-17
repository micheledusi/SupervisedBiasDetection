# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module implements the AbstractClassifier class with a Random Forest classifier.
# The Random Forest is implemented in the scikit-learn library.

from sklearn.ensemble import RandomForestClassifier
import torch

from model.classification.base import AbstractClassifier, ClassesDict
from model.classification.svm import IndexedClassesDict


class ForestClassifier(AbstractClassifier):
	"""
	This class implements a classifier with a DecisionTree model (from scikit-learn library).
	The model is trained with the 'train' method, and it can be used to predict the class of a new sample with the 'evaluate' method.
	"""
	def __init__(self) -> None:
		super().__init__()
		self._model = RandomForestClassifier(criterion='gini')

	@property
	def features_relevance(self) -> torch.Tensor:
		return torch.Tensor(self._model.feature_importances_).squeeze()

	def _fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
		self._model.fit(x, y.ravel())

	def _predict(self, x: torch.Tensor) -> torch.Tensor:
		return torch.Tensor(self._model.predict(x))

	def _compute_class_tensors(self, values: list[str]) -> ClassesDict:
		labels: list[str] = sorted(set(values))
		return IndexedClassesDict(labels)

