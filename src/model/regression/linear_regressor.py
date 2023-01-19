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
from utility.cache import CacheManager


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

        values_set: list[str] = sorted(set(dataset['value']))   # In this way, the order of the values is computed according to the alphabetical order
        features = Features({'label': ClassLabel(num_classes=len(values_set), names=values_set)})
        labels_ds = Dataset.from_dict({"label": dataset['value']}, features=features) 
        labels_ds = labels_ds.with_format("torch", device=torch.device)
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
        def predict_fn(sample):
            inputs = Variable(sample['embedding'])
            sample['prediction'] = self.model(inputs)
            return sample
        return dataset.map(predict_fn, batched=True)


    @property
    def features_relevance(self) -> torch.Tensor:
        return self.model.weights



if __name__ == '__main__':
    torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    tested_property = 'quality'
    input_words_file = 'data/stereotyped-p/quality/words-01.csv'
    input_templates_file = 'data/stereotyped-p/quality/templates-01.csv'
    param_select_templates = 'all'
    param_average_templates = True
    param_average_tokens = True
    
    # Disk management for embedding datasets
    cacher = CacheManager()
    name = 'quality_words'
    group = 'embedding'
    metadata = {
        'stereotyped_property': tested_property,
        'input_words': input_words_file,
        'input_templates': input_templates_file,
        'select_templates': param_select_templates,
        'average_templates': param_average_templates,
        'average_tokens': param_average_tokens,
    }

    if cacher.exists(name, group, metadata):
        print("Loading the embedding dataset from the cache...")
        embedding_dataset = cacher.load(name, group, metadata)
    else:
        print("Embedding the words...")
        from model.embedding.word_embedder import WordEmbedder
        # Loading the datasets
        templates: Dataset = Dataset.from_csv('data/stereotyped-p/quality/templates-01.csv')
        words: Dataset = Dataset.from_csv('data/stereotyped-p/quality/words-01.csv').shuffle(seed=42)
        # Creating the word embedder
        word_embedder = WordEmbedder(select_templates='all', average_templates=True, average_tokens=True)
        # Embedding a word
        embedding_dataset = word_embedder.embed(words, templates)
        # Caching the embedding dataset
        cacher.save(embedding_dataset, name, group, metadata)

    # Squeezing the embeddings to remove the template dimension and the token dimension
    def squeeze_fn(sample):
        sample['embedding'] = sample['embedding'].squeeze()
        return sample
    embedding_dataset = embedding_dataset.map(squeeze_fn, batched=True)

    # Splitting the dataset into train and test
    embedding_dataset = embedding_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

    # Using the embeddings to train the model
    reg_model = LinearRegressor()
    reg_model.train(embedding_dataset['train'])

    # Predict the values
    results = reg_model.predict(embedding_dataset['test'])
    score: int = 0
    for result in results:
        predicted_value = 'negative' if result['prediction'] < 0.5 else 'positive'
        guessed = predicted_value == result['value']
        print(f"{result['word']:20s} => ", guessed, f"({predicted_value} vs {result['value']})")
        score += 1 if guessed else 0
    print(f"Score: {score}/{len(results)} ({score/len(results)*100:.2f}%)")