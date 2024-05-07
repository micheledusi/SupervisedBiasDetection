# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module contains the classes that perform a linear classification.
# The 'LinearClassifier' class extends the 'AbstractClassifier' class, 
# thus it can be used as a classifier in other modules of the project.
# The 'TorchClassifier' class implements the core model of the linear classifier,
# with a linear layer and a softmax function.


import logging
import torch
from torch.autograd import Variable

from model.classification.base import AbstractClassifier, ClassesDict
from utils.const import DEVICE

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class TorchClassifier(torch.nn.Module):
	"""
	This class implements a linear classification model in PyTorch.

	The model is a linear layer, followed by a softmax function. 
	Both the linear layer and the softmax function are implemented in PyTorch.

	This class involves only the core model. For that, it is not meant to be used directly.
	To properly use it, the LinearClassifier class is provided as a wrapper.
	"""
	def __init__(self, in_features: int, out_features: int) -> None:
		super(TorchClassifier, self).__init__()
		self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
		self.softmax = torch.nn.Softmax(dim=-1)
		# Setting device
		if torch.cuda.is_available():
			self.cuda()
		else:
			self.cpu()

	def forward(self, x):
		x.to(DEVICE)
		out = self.linear(x)
		out = self.softmax(out)
		return out

	@property
	def weights(self) -> torch.Tensor:
		"""
		This method returns the weights of the linear layer, representing the weights of the linear classification model.
		The weights are returned as a torch.Tensor object of shape [#out_features, #in_features].

		:return: The weights of the linear classification model, as a torch.Tensor object.
		"""
		return self.linear.weight


class OneHotClassesDict(ClassesDict):
	"""
	This class represents a dictionary where for each label corresponds a one-hot representation.
	"""
	def __init__(self, labels: list[str] | tuple[str]) -> None:
		# Calling the superclass constructor with the labels
		super().__init__(labels)
		# Initializing the dictionary
		base = torch.eye(len(self.labels))
		self._label2tensor = {label: base[i] for i, label in enumerate(self.labels)}

	def get_tensor(self, value: str) -> torch.Tensor:
		return self._label2tensor[value]

	def get_label(self, tensor: torch.Tensor) -> str:
		index = torch.argmax(tensor)
		return self._labels[index]


class LinearClassifier(AbstractClassifier):
	"""
	This class represents a classifier for embeddings.
	The classification core model is implemented in the TorchClassifier class, while this class is a wrapper for it.
	This class also implements the training and the prediction of the model. Plus, it is a subclass of the AbstractClassifier interface.
	This means that it can be used in other parts of the project, as a classifier for embeddings.

	The classification involves an embedding of a word as the input (independent variable), and a protected property value as the output (dependent variable).
	"""
	learning_rate: float = 0.01
	epochs: int = 1000
	
	def __init__(self) -> None:
		super().__init__()
		self.model = None

	@property
	def features_relevance(self) -> torch.Tensor:
		linear_classifier_weights = self.model.weights.abs()
		features_relevance = torch.mean(linear_classifier_weights, dim=0)
		return features_relevance
	
	def _fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
		# Debugging information
		logger.debug(f"X shape: {x.shape}")
		logger.debug(f"Y shape: {y.shape}")
		logger.debug(f"Y squeezed shape: {y.squeeze().shape}")
		# If the input tensor is 1-dimensional, we assume that the N° of features is 1. 
  		# Thus, we need to add a dimension to it in order to fit the model
		if len(x.shape) == 1:
			x = x.unsqueeze(1)
			logger.debug(f"Unsqueezing X shape: {x.shape} for training")

		# Define the model
		input_size: int = x.shape[1]
		output_size: int = y.shape[1]
		self.model = TorchClassifier(in_features=input_size, out_features=output_size)

		# Define the loss function and the optimizer
		criterion = torch.nn.MSELoss() 
		optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

		# Prepare the data for the training
		# The inputs are the features considered in the classification
		x_var = Variable(x)
		# The labels are the values we want to learn and predict
		y_var = Variable(y)

		for epoch in range(self.epochs):
			# Clear gradient buffers, because we don't want any gradient from previous epoch to carry forward, dont want to cumulate gradients
			optimizer.zero_grad()
			# Compute the effective output of the model from the input
			outputs = self.model(x_var)
			# Comparing the predicted output with the actual output, we compute the loss
			loss = criterion(outputs, y_var)
			# Computing the gradients from the loss w.r.t. the parameters
			loss.backward()
			# Updating the parameters
			optimizer.step()
			# if epoch % 100 == 0:
			#	print('Epoch {} => loss = {}'.format(epoch, loss.item()))

	def _predict(self, x: torch.Tensor) -> torch.Tensor:
		# If the input tensor is 1-dimensional, we assume that the N° of features is 1. Thus, we need to add a dimension to it in order to fit the model
		if len(x.shape) == 1:
			x = x.unsqueeze(1)
			logger.debug(f"Unsqueezing X shape: {x.shape} for prediction")

		x_var = Variable(x)
		outputs = self.model(x_var)
		return outputs

	def _compute_class_tensors(self, values: list[str]) -> ClassesDict:
		"""
		This method returns a dictionary where for each label corresponds a one-hot representation.
		The values for the labels are taken from the given list of labels.
		If two values are equal, then the corresponding one-hot representations are equal.

		The one-hot representation is a torch.Tensor object of shape [#classes].
		Each element of the tensor is 0, except for the element corresponding to the label of the input, which is 1.

		That is useful to perform a classification task, where the output of the model is a vector of probabilities (given by the softmax function)
		and the label of the input is the index of the element with the highest probability.

		:param labels: The list of labels to consider.
		:return: The dictionary of one-hot representations.
		"""
		labels: list[str] = sorted(set(values))
		return OneHotClassesDict(labels)
