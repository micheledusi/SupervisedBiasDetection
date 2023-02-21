# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module implements the AbstractClassifier class with a SVM classifier.
# The SVM is implemented in the scikit-learn library.

from sklearn.svm import LinearSVC
import torch

from model.classification.base import AbstractClassifier, ClassesDict


class BinaryClassesDict(ClassesDict):
	"""
	This class implements a dictionary of classes, with two classes.
	The classes are associated to a tensor of shape [1], with values -1 and +1.
	"""
	def __init__(self, labels: list[str] | tuple[str]) -> None:
		super().__init__(labels)
		self._classes = {
			self.labels[0]: torch.Tensor([-1]), 
			self.labels[1]: torch.Tensor([+1]),
			}

	def get_tensor(self, value: str) -> torch.Tensor:
		return self._classes[value]

	def get_label(self, tensor: torch.Tensor) -> str:
		if tensor.item() <= 0:
			return self.labels[0]
		else:
			return self.labels[1]


class SVMClassifier(AbstractClassifier):
	"""
	This class implements a classifier with a SVM model (from scikit-learn library).
	The model is trained with the 'train' method, and it can be used to predict the class of a new sample with the 'evaluate' method.
	"""
	def __init__(self) -> None:
		super().__init__()
		self._model = LinearSVC(penalty="l2", loss="squared_hinge", dual=False)

	@property
	def features_relevance(self) -> torch.Tensor:
		return torch.Tensor(self._model.coef_).squeeze().abs()

	def _fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
		self._model.fit(x, y.ravel())

	def _predict(self, x: torch.Tensor) -> torch.Tensor:
		return torch.Tensor(self._model.predict(x))

	def _compute_class_tensors(self, values: list[str]) -> ClassesDict:
		labels: list[str] = sorted(set(values))
		return BinaryClassesDict(labels)

