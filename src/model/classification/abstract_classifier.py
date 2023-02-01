# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
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


class AbstractClassifier(ABC):
    """
    This class represents a classifier performing a classification task.
    The regression involves an embedding of a word as the input (independent variable), and a protected property value as the output (dependent variable).

    The class is abstract, and it is meant to be extended by concrete classes implementing the classification task
    according to different approaches (e.g. linear classification, neural networks, etc.).

    This model is intended to be trained on a dataset of words, for which we know the value of the protected property.
    E.g. for the <gender> protected property, we should provide the embedding of words like "he" and "she", 
    along with their protected values, "male" and "female" respectively.
    """

    def __init__(self):
        pass

    @abstractmethod
    def train(self, dataset: Dataset) -> None:
        """
        This method trains the model on a dataset of words, for which we know the value of the protected property.
        The dataset, a Dataset object, must contain the following columns:
        - "word": the word, as a string
        - "embedding": the embedding of the word, as a torch.Tensor object
        - "value": the value of the protected property, as a string

        :param dataset: A dataset of words, as a Dataset object. 
        """
        raise NotImplementedError("This method must be implemented by the subclasses.")

    @abstractmethod
    def predict(self, dataset: Dataset) -> Dataset:
        """
        This method predicts the value of the protected property, given the embedding of a word.

        :param dataset: A list of embeddings of words, as a Dataset object. The dataset must contain a column named "embedding" and a column named "word".
        :return: A dataset with the same structure of the input dataset, but with an additional column named "prediction" containing the predicted value of the protected property.
        """
        raise NotImplementedError("This method must be implemented by the subclasses.")

    @abstractproperty
    def features_relevance() -> torch.Tensor:
        """
        This method returns, for each feature (i.e. each embedding dimension), the "importance" of the feature for the prediction.
        The "importance" measures how much the feature contributes to the prediction, and it can be defined in different ways according to the approach used.

        Note: to get the importance of the features, the model must be trained.

        :return: A tensor containing the importance of each feature, with the same shape as the embedding.
        """
        raise NotImplementedError("This method must be implemented by the subclasses.")
