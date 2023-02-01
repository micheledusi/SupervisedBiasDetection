# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module contains the classes that perform a linear regression.
# The core model for the linear regression in PyTorch is implemented 
# in the TorchLinearRegression class, using the PyTorch Linear module.
# The LinearRegression class is then wrapped in the LinearRegressor class, 
# which extends the _AbstractRegressor interface.


import torch
from torch.autograd import Variable
from datasets import Dataset
import sys
from pathlib import Path

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.parent))
from model.classification.abstract_classifier import AbstractClassifier
from data_processing.sentence_maker import PP_PATTERN, SP_PATTERN
from utils.const import DEVICE
from utils.cache import get_cached_embeddings


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
		# out = self.softmax(out)
		return out

	@property
	def weights(self) -> torch.Tensor:
		"""
		This method returns the weights of the linear layer, representing the weights of the linear classification model.
		The weights are returned as a torch.Tensor object of shape [#out_features, #in_features].

		:return: The weights of the linear classification model, as a torch.Tensor object.
		"""
		return self.linear.weight


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

	LabelsDict = dict[str, torch.Tensor]
	
	def __init__(self) -> None:
		super().__init__()

	@staticmethod
	def define_class_labels(dataset: Dataset, column: str = 'value') -> LabelsDict:
		"""
		This method returns a dictionary where for each label corresponds a numerical representation.
		The values for the labels are taken from the given dataset, by default from the 'value' column.
		If two values are equal, then the corresponding "numerical representations" are equal.

		The numerical representation is a one-hot vector, where the i-th element is 1 if the label is the i-th label in the list of labels, and 0 otherwise.
		That is useful to perform the classification task, since the output of the model is a vector of probabilities (given by the softmax function),
		and the label is the index of the maximum probability.

		:param dataset: The dataset to analyze.
		:param column: The column name of the dataset to use to extract the labels. Default: 'value'.
		:return: The list of labels of the dataset.
		"""
		values_list = sorted(set(dataset[column]))
		if len(values_list) == 1:
			raise ValueError("The dataset contains only one class value. We cannot perform a classification task.")
		base = torch.eye(len(values_list))
		labels = {value: base[i] for i, value in enumerate(values_list)}
		return labels
	
	@staticmethod
	def _extract_labels(values: list[str], labels: LabelsDict) -> torch.Tensor:
		return torch.Tensor([labels[value] for value in values])

	def train(self, dataset: Dataset) -> LabelsDict:
		# Define the model
		embeddings = dataset['embedding']	# Embeddings are assumed to be a list of tensors with the same size: list( torch.Tensor[#features] )
		self.input_size = embeddings[0].shape[0]
		# We define the output size as the number of distinct values in the dataset
		class_labels = LinearClassifier.define_class_labels(dataset)
		self.output_size = len(class_labels)
		# Define the model
		self.model = TorchClassifier(in_features=self.input_size, out_features=self.output_size)

		# Define the loss function and the optimizer
		criterion = torch.nn.MSELoss() 
		optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

		# Prepare the data for the training
		# The inputs are the features considered in the regression
		inputs = Variable(embeddings)
		# The labels are the values we want to learn and predict
		labels = Variable(torch.stack([class_labels[value] for value in dataset['value']]))

		for epoch in range(self.epochs):
			# Clear gradient buffers, because we don't want any gradient from previous epoch to carry forward, dont want to cumulate gradients
			optimizer.zero_grad()
			# Compute the effective output of the model from the input
			outputs = self.model(inputs)
			# Comparing the predicted output with the actual output, we compute the loss
			loss = criterion(outputs, labels)
			# Computing the gradients from the loss w.r.t. the parameters
			loss.backward()
			# Updating the parameters
			optimizer.step()
			if epoch % 100 == 0:
				print('Epoch {} => loss = {}'.format(epoch, loss.item()))
		return class_labels
	
	def predict(self, dataset: Dataset) -> Dataset:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		def predict_fn(sample):
			embedding = sample['embedding'].to(device)
			inputs = Variable(embedding)
			sample['prediction'] = self.model(inputs)
			return sample
		return dataset.map(predict_fn, batched=True)

	@property
	def features_relevance(self) -> torch.Tensor:
		"""
		This method returns the relevance of each features in the linear classification model.

		The original weights of the classifier are a torch.Tensor object of shape [#classes, #in_features], where #classes is the number of classes to predict.
		Since we're interested in the relevance of the features, we compute the mean of the weights for each feature, 
		and return the result as a torch.Tensor object of shape [#in_features].
		(In our case, the number of features is 768).

		:return: The relevance of each features in the linear classification model.
		"""
		linear_classifier_weights = self.model.weights
		features_relevance = torch.mean(linear_classifier_weights, dim=0)
		return features_relevance


if __name__ == '__main__':

	property = 'religion'
	property_type = 'protected' # "stereotyped", "protected"

	pattern = SP_PATTERN if property_type == 'stereotyped' else PP_PATTERN if property_type == 'protected' else None
	words_file = f'data/{property_type}-p/{property}/words-02.csv'
	templates_file = f'data/{property_type}-p/{property}/templates-01.csv'
	embedding_dataset = get_cached_embeddings(property, pattern, words_file, templates_file, rebuild=True)

	def print_dataset_info(dataset: Dataset):
		# Printing the number of samples in the dataset
		print("Number of samples: ", len(dataset))
		# Printing the number of samples for each value (i.e., for each class) in the 'value' column
		classes = set(dataset['value'])
		print("Number of different classes: ", len(classes))
		# For each value (i.e., for each class) in the 'value' column, printing the number of samples
		for value in classes:
			num_samples = len(list(filter(lambda x: x == value, dataset['value'])))
			print(f"\tNumber of samples for class '{value}': ", num_samples)

	# Squeezing the embeddings to remove the template dimension and the token dimension
	def squeeze_fn(sample):
		sample['embedding'] = sample['embedding'].squeeze()
		return sample
	embedding_dataset = embedding_dataset.map(squeeze_fn, batched=True)

	# Splitting the dataset into train and test
	embedding_dataset = embedding_dataset.train_test_split(test_size=0.5, shuffle=True, seed=42)

	# Using the embeddings to train the model
	reg_model = LinearClassifier()
	class_labels = reg_model.train(embedding_dataset['train'])
	print_dataset_info(embedding_dataset['train'])

	# Predict the values
	results = reg_model.predict(embedding_dataset['test'])

	# Print the results
	guesses = 0
	for result in results:
		predicted_value = torch.argmax(result['prediction']).item()
		actual_value = torch.argmax(class_labels[result['value']]).item()
		print(f"Word: {result['word']:20s}", end=' ')
		print(f"Predicted class:", predicted_value, end='\t')
		print(f"Actual class:", actual_value, end='\t')
		if predicted_value == actual_value:
			guesses += 1
			print()
		else:
			print(" WRONG")

	print(f"Validation accuracy: {guesses}/{len(results)} ({guesses/len(results)*100:.2f}%)")

	# Print the relevance of each feature
	features_relevance = reg_model.features_relevance
	print("Features relevance dimensions: ", features_relevance.shape)