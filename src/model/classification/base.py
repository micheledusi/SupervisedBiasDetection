# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module contains a class representing a classifier, that is, a model that can be trained and used to predict a variable value.
# In our case, the model infers the value of a protected property from the embedding of a word (e.g. the vector in output to the NLP model).
# 
# An example of classificaiton used in our project is the one that predicts the <gender> of a word from its embedding:
# The <gender> is a protected property, for which we consider the values "male" and "female" among the possible values.
# After training the model on a dataset of words, for which we know the <gender> value, we can use the model to predict
# the <gender> value of new words, given their embedding.


from abc import ABC, abstractmethod, abstractproperty
from datasets import Dataset
import torch


class ClassesDict():
	"""
	This class represents a dictionary mapping the property values to the corresponding classifier outputs (i.e. tensors) used in the model.
	The mapping is bidirectional: from the property value to the output tensor, and viceversa.
	However, while the mapping from the property value to the output tensor is exact, the mapping from the output tensor to the property value is approximate.
	Each class has a unique output tensor, but a given output tensor may not have a perfect match in the dictionary, so the method returns the closest match.
	"""
	def __init__(self):
		pass

	@abstractproperty
	def labels(self) -> tuple[str]:
		"""
		This method returns the property values in the dictionary.
		The collection is returned as a tuple of strings.

		:return: The tuple of property values in the dictionary.
		"""
		raise NotImplementedError("This method must be implemented by the subclasses.")

	@abstractmethod
	def get_tensor(self, value: str) -> torch.Tensor:
		"""
		This method returns the output tensor corresponding to the given property label.
		This is an exact method: each class has a unique output tensor.
		If the given property label is not in the dictionary, a KeyError is raised.

		:param value: The property class label.
		:return: The corresponding output tensor.
		"""
		raise NotImplementedError("This method must be implemented by the subclasses.")
	
	@abstractmethod
	def get_label(self, output: torch.Tensor) -> str:
		"""
		This method returns the property value corresponding to the given output tensor.
		This is an approximate method: a given output tensor may not have a perfect match in the dictionary, so the method returns the closest match.

		:param output: The output tensor.
		:return: The property value corresponding to the given output tensor.
		"""
		raise NotImplementedError("This method must be implemented by the subclasses.")

	def __getitem__(self, key: str | torch.Tensor) -> torch.Tensor | str:
		"""
		This method returns the output tensor corresponding to the given property label, or the property value corresponding to the given output tensor.
		The choice is made according to the type of the given key.

		:param key: The property label or the output tensor.
		:return: The corresponding output tensor or the property value.
		"""
		if isinstance(key, str):
			return self.get_tensor(key)
		elif isinstance(key, torch.Tensor):
			return self.get_label(key)
		else:
			raise TypeError(f"Invalid key type: {type(key)}.")

	def __contains__(self, key: str) -> bool:
		"""
		This method checks if the given key is in the dictionary.
		Since a tensor cannot be used as a key, the method checks if the given key is a property label.

		:param key: The property label.
		:return: True if the key is in the dictionary, False otherwise.
		"""
		return key in self.labels

	def __len__(self) -> int:
		"""
		This method returns the number of classes in the dictionary.

		:return: The number of classes in the dictionary.
		"""
		return len(self.labels)


class AbstractClassifier(ABC):
	"""
	This class represents a classifier performing a classification task.
	The classification involves an embedding of a word as the input (independent variable), and a protected property value as the output (dependent variable).

	The class is abstract, and it is meant to be extended by concrete classes implementing the classification task
	according to different approaches (e.g. linear classification, support vector machines, neural networks, etc.).

	This model is intended to be trained on a dataset of words, for which we know the value of the protected property.
	E.g. for the <gender> protected property, we should provide the embedding of words like "he" and "she", 
	along with their protected values, "male" and "female" respectively.
	"""
	def __init__(self):
		self._classes: ClassesDict = None
		pass

	@abstractproperty
	def features_relevance(self) -> torch.Tensor:
		"""
		This method returns, for each feature (i.e. each embedding dimension), the "importance" of the feature for the prediction.
		The "importance" measures how much the feature contributes to the prediction, and it can be defined in different ways according to the approach used.

		Note: to get the importance of the features, the model must be trained.

		:return: A tensor containing the importance of each feature, with the same shape as the embedding.
		"""
		raise NotImplementedError("This method must be implemented by the subclasses.")
	
	@property
	def classes(self) -> ClassesDict:
		"""
		This method returns the dictionary mapping the property values to the corresponding labels used in the model.
		"""
		if not self._classes:
			raise ValueError("The model must be trained before accessing the classes.")
		return self._classes

	@abstractmethod
	def _fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
		"""
		This method fits the model on the given data.
		It is meant to be implemented by the subclasses, and it is called by the train method.
		"""
		raise NotImplementedError("This method must be implemented by the subclasses.")
	
	@abstractmethod
	def _predict(self, x: torch.Tensor) -> torch.Tensor:
		"""
		This method uses the trained classifier to predict the outcome, given the input.
		In our case, the input is the embedding of a word, and the output is the value of the protected property.
		It is meant to be implemented by the subclasses, and it is called by the predict method.
		"""
		raise NotImplementedError("This method must be implemented by the subclasses.")
		
	@abstractmethod
	def _compute_class_tensors(self, labels: list[str]) -> ClassesDict:
		"""
		This method computes the tensors corresponding to the labels for each class of the property.
		It is meant to be implemented by the subclasses, and it is called just before training the model.
		After that, the tensors are stored in the _classes attribute, and they can be accessed by the `classes` property.
		"""
		raise NotImplementedError("This method must be implemented by the subclasses.")
	
	def __map_values_to_classes(self, values: list[str]) -> torch.Tensor:
		"""
		This method maps the values of the property to the corresponding labels used in the model.
		"""
		if not self._classes:
			raise ValueError("Incorrect order of calls: the classes dictionary is not instantiated yet.")
		classes_ref: ClassesDict = self._classes
		return torch.stack([classes_ref[value] for value in values])

	def train(self, dataset: Dataset, input_column: str='embedding', output_column: str='value') -> None:
		"""
		This method trains the model on a dataset of words, for which we know the value of the property.
		The dataset must contain a column for the input values, and a column for the output values.
		By default, the input column is named "embedding" and the output column is named "value", but these names can be changed by the user.

		:param dataset: A dataset of words, as a Dataset object.
		:param input_column: The name of the column containing the input values. Default: "embedding".
		:param output_column: The name of the column containing the output values. Default: "value".
		"""
		# Compute the tensors containing the labels for each class of the property.
		inputs = dataset[input_column]
		if isinstance(inputs, list):
			inputs = torch.stack(inputs)
		self._classes = self._compute_class_tensors(dataset[output_column])
		outputs = self.__map_values_to_classes(dataset[output_column])

		assert inputs.shape[0] == outputs.shape[0], "The number of input values must be equal to the number of output values."

		# Calling the sub-method to perform the training.
		self._fit(inputs, outputs)

	def evaluate(self, dataset: Dataset, input_column: str='embedding') -> Dataset:
		"""
		This method predicts the value of the protected property, given the embedding of a word.

		:param dataset: A list of embeddings of words, as a Dataset object. The dataset must contain a column named "embedding" and a column named "word".
		:return: A dataset with the same structure of the input dataset, but with an additional column named "prediction" containing the predicted value of the protected property.
		"""
		# Predict the outcome for each embedding in the dataset.
		inputs = dataset[input_column]
		if isinstance(inputs, list):
			inputs = torch.stack(inputs)
		
		# Calling the sub-method to perform the prediction.
		predictions = self._predict(inputs).tolist()

		# Add a column to the dataset containing the predictions.
		return dataset.add_column(name="prediction", column=predictions).with_format("pytorch")

