# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
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
    epochs: int = 100
    
    def __init__(self) -> None:
        super().__init__()
        self.model = TorchLinearRegression(in_features=self.input_size, out_features=self.output_size)
        if torch.cuda.is_available():
            self.model.to("cuda")

    
    def train(self, dataset: Dataset) -> None:
        # Define the loss function and the optimizer
        criterion = torch.nn.MSELoss() 
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # Prepare the data stacking a list of tensors into a single tensor
        inputs = Variable(dataset['embedding'])
        if torch.cuda.is_available():
            inputs.to("cuda")

        values_set: set[str] = set(dataset['value'])
        features = Features({'label': ClassLabel(num_classes=len(values_set), names=list(values_set))})
        labels_ds = Dataset.from_dict({"label": dataset['value']}, features=features) 
        labels_ds = labels_ds.with_format("torch", device=torch.device)
        labels = Variable(labels_ds['label'].unsqueeze(1).float())

        print(labels.data)
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

            print('Epoch {} => loss = {}'.format(epoch, loss.item()))
    

    def predict(self, dataset: Dataset) -> Dataset:
        def map_fn(example):
            inputs = Variable(example['embedding'])
            example['prediction'] = self.model(inputs)
            return example
        return dataset.map(map_fn, batched=True)


    @property
    def features_relevance(self) -> torch.Tensor:
        return self.model.weights



if __name__ == '__main__':
    import numpy as np

    torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Testing the linear regression model

    # Prepare the training dataset
    ds: Dataset = Dataset.from_dict({'word': [], 'embedding': [], 'value': []})
    for i in range(500):
        # (1) sample
        fake_embedding = np.random.rand(LinearRegressor.input_size)
        fake_embedding[0] = 1.0
        fake_embedding[1] = 1.0
        fake_embedding[2] = 1.0
        fake_value = 'a'
        ds = ds.add_item({'word': "AAA", 'embedding': fake_embedding, 'value': fake_value})
        # (0) sample
        fake_embedding = np.random.rand(LinearRegressor.input_size)
        fake_embedding[0] = 0.0
        fake_embedding[1] = 0.0
        fake_embedding[2] = 0.0
        fake_value = 'b'
        ds = ds.add_item({'word': "BBB", 'embedding': fake_embedding, 'value': fake_value})
    ds = ds.with_format("torch")
    print("Training dataset: ", ds)
    print("First element: ", ds[0])
    print("First element embedding size: ", ds[0]['embedding'].size())

    print("Embedding column: ", ds['embedding'])
    print("It's a tensor? >> ", isinstance(ds['embedding'], torch.Tensor))
    
    # Train the model
    reg_model = LinearRegressor()
    reg_model.train(ds)

    # Prepare the test dataset
    ds: Dataset = Dataset.from_dict({'word': [], 'embedding': []})
    for i in range(2):
        # (1) sample
        fake_embedding = np.random.rand(LinearRegressor.input_size)
        fake_embedding[0] = 2.0
        fake_embedding[1] = 2.0
        fake_embedding[2] = 2.0
        ds = ds.add_item({'word': "AAA", 'embedding': fake_embedding})
        # (0) sample
        fake_embedding = np.random.rand(LinearRegressor.input_size)
        fake_embedding[0] = 0.0
        fake_embedding[1] = 0.0
        fake_embedding[2] = 0.0
        ds = ds.add_item({'word': "BBB", 'embedding': fake_embedding})
    ds = ds.with_format("torch")
    print("Test dataset: ", ds)

    # Predict the values
    ds = reg_model.predict(ds)
    print("Predictions: ", ds['prediction'])

    # Print the weights
    print("What are the weights? ", reg_model.features_relevance)