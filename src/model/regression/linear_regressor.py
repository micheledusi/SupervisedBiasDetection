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
from datasets import ClassLabel, Dataset, Features
import sys
from pathlib import Path

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.parent))
from model.regression.abstract_regressor import AbstractRegressor
from data_processing.sentence_maker import SP_PATTERN
from utility.cache_embedding import get_cached_embeddings


class TorchLinearRegression(torch.nn.Module):
	"""
	This class implements a linear regression model using the PyTorch Linear module.
	"""
	def __init__(self, in_features: int, out_features: int) -> None:
		super(TorchLinearRegression, self).__init__()
		self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features)

	def forward(self, x):
		out = self.linear(x)
		return out

	@property
	def weights(self) -> torch.Tensor:
		"""
		This method returns the weights of the linear regression model.

		:return: The weights of the linear regression model, as a torch.Tensor object.
		"""
		return self.linear.weight


class LinearRegressor(AbstractRegressor):
	"""
	This class represents a regressor performing a regression task using a linear SVM.
	The regression involves an embedding of a word as the input (independent variable), and a protected property value as the output (dependent variable).
	"""
	input_size: int = 768
	output_size: int = 1
	learning_rate: float = 0.005
	epochs: int = 1000
	
	def __init__(self) -> None:
		super().__init__()
		self.model = TorchLinearRegression(in_features=self.input_size, out_features=self.output_size)

	def train(self, dataset: Dataset) -> None:
		# Define the loss function and the optimizer
		criterion = torch.nn.MSELoss() 
		optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

		# Prepare the data stacking a list of tensors into a single tensor
		inputs = Variable(dataset['embedding'])

		values_set: list[str] = sorted(set(dataset['value']))   # In this way, the order of the values is computed according to the alphabetical order
		features = Features({'label': ClassLabel(num_classes=len(values_set), names=values_set)})
		labels_ds = Dataset.from_dict({"label": dataset['value']}, features=features) 
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		labels_ds = labels_ds.with_format("torch", device=device)
		labels = Variable(labels_ds['label'].unsqueeze(1).float())
		# TODO: Convert labels to -1 and +1

		# Note: we add the unsqueeze(1) to convert the tensor from shape (batch_size,) to (batch_size, 1)

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
			# print('Epoch {} => loss = {}'.format(epoch, loss.item()))
	
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
		return self.model.weights


if __name__ == '__main__':
	
	stereotyped_property = 'quality'
	stereotyped_words_file = f'data/stereotyped-p/{stereotyped_property}/words-01.csv'
	stereotyped_templates_file = f'data/stereotyped-p/{stereotyped_property}/templates-01.csv'
	stereotyped_embedding_dataset = get_cached_embeddings(stereotyped_property, SP_PATTERN, stereotyped_words_file, stereotyped_templates_file)

	# Squeezing the embeddings to remove the template dimension and the token dimension
	def squeeze_fn(sample):
		sample['embedding'] = sample['embedding'].squeeze()
		return sample
	embedding_dataset = stereotyped_embedding_dataset.map(squeeze_fn, batched=True)

	# Splitting the dataset into train and test
	embedding_dataset = embedding_dataset.train_test_split(test_size=0.5, shuffle=True, seed=42)

	# Using the embeddings to train the model
	reg_model = LinearRegressor()
	reg_model.train(embedding_dataset['train'])

	# Predict the values
	results = reg_model.predict(embedding_dataset['test'])
	score: int = 0
	for result in results:
		predicted_value = 'negative' if result['prediction'] < 0.5 else 'positive'
		guessed = predicted_value == result['value']
		print(f"{result['word']:20s} => {str(guessed):5s}", f"(Predicted: {predicted_value}, but expected: {result['value']})")
		score += 1 if guessed else 0
	print(f"Score: {score}/{len(results)} ({score/len(results)*100:.2f}%)")