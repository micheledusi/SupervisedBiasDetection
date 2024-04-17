# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module implements the AbstractClassifier class with a SVM classifier.
# The SVM is implemented in the scikit-learn library.

import logging
from sklearn.svm import LinearSVC
import torch

from model.classification.base import AbstractClassifier, ClassesDict

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class BinaryClassesDict(ClassesDict):
	"""
	This class implements a dictionary of classes, with two classes.
	The classes are associated to a tensor of shape [1], with values -1 and +1.
	"""
	def __init__(self, labels: list[str] | tuple[str]) -> None:
		super().__init__(labels)
		if len(self.labels) != 2:
			raise ValueError("The number of classes must be 2. Please, use the IndexedClassesDict class instead.")
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


class IndexedClassesDict(ClassesDict):
	"""
	This class implements a dictionary of classes, with a fixed number of classes.
	The classes are associated to a tensor of shape [1], with values 0, 1, 2, ..., #classes-1.
	"""
	def __init__(self, labels: list[str] | tuple[str]) -> None:
		super().__init__(labels)
		self._classes = {label: torch.Tensor([i]) for i, label in enumerate(self.labels)}

	def get_tensor(self, value: str) -> torch.Tensor:
		return self._classes[value]

	def get_label(self, tensor: torch.Tensor) -> str:
		return self.labels[int(tensor.item())]


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
		features_relevance =  torch.Tensor(self._model.coef_).squeeze().abs()
		if len(features_relevance.shape) == 2:
			features_relevance = features_relevance.mean(dim=0)
		return features_relevance

	def _fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
		logger.debug("X shape:", x.shape)
		logger.debug("Y shape:", y.shape)
		logger.debug("Y squeezed shape:", y.squeeze().shape)
		self._model.fit(x, y.squeeze())

	def _predict(self, x: torch.Tensor) -> torch.Tensor:
		return torch.Tensor(self._model.predict(x))

	def _compute_class_tensors(self, values: list[str]) -> ClassesDict:
		labels: list[str] = sorted(set(values))
		if len(labels) == 2:
			return BinaryClassesDict(labels)
		else:
			return IndexedClassesDict(labels)

