# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#							    #
#   Author:  Michele Dusi	    #
#   Date:	 2024			    #
# - - - - - - - - - - - - - - - #

# This module contains a class for normalization of relevance scores.
# Normalization is the process of transforming the raw relevance scores into normalized relevance scores.
# Normalization can be done in different ways, depending on the specific needs of the project.

import logging
from typing import Callable
import torch


# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class RelevanceNormalizer:
    """
    A generic relevance normalizer, with several methods to normalize relevance scores.
    """

    DEFAULT_MIN = 0
    DEFAULT_MAX = 1
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_AMPLITUDE = 10

    def __init__(self, relevance_scores: torch.Tensor):
        """
        Initializer for the relevance normalizer.
        """
        logger.info("Initializing a relevance normalizer.")
        relevance_scores = relevance_scores.squeeze()
        if len(relevance_scores.shape) != 1:
            raise ValueError("Relevance scores must be a 1-dimensional tensor.")
        self.__scores = relevance_scores
        self.__normalized_scores: dict[str, torch.Tensor] = None

    
    @property
    def scores(self) -> torch.Tensor:
        """
        The raw relevance scores to normalize.
        """
        return self.__scores
    

    def __retrieve(self, normalization_name: str, normalization_fun: Callable) -> torch.Tensor:
        """
        Retrieves the normalized relevance scores, if they have already been computed.
        Otherwise, it computes them using the provided normalization function.

        :param normalization_name: The name of the normalization method.
        :param normalization_fun: The normalization function to apply.
        :return: The normalized relevance scores.
        """
        if self.__normalized_scores is None:
            self.__normalized_scores = {}
        if normalization_name not in self.__normalized_scores:
            logger.info("Normalizing relevance scores with method \"%s\".", normalization_name)
            self.__normalized_scores[normalization_name] = normalization_fun()
        return self.__normalized_scores[normalization_name]


    def linear(self, min: float = DEFAULT_MIN, max: float = DEFAULT_MAX) -> torch.Tensor:
        """
        Linear normalization of the relevance scores.
        The normalization is done by scaling the relevance scores linearly
        between the minimum and maximum values provided (default: 0 and 1).

        :param min: The minimum value for the normalization.
        :param max: The maximum value for the normalization.
        :return: The normalized relevance scores.
        """
        def normalization() -> torch.Tensor:
            min_score = torch.min(self.scores)
            max_score = torch.max(self.scores)
            return (self.scores - min_score) / (max_score - min_score) * (max - min) + min
        return self.__retrieve(f"linear_{min}_{max}", normalization)
    

    def linear_opposite(self) -> torch.Tensor:
        """
        Linear normalization of the relevance scores, with opposite values.
        The normalization is done by scaling the relevance scores linearly
        between 1 and 0, with the opposite values.

        :return: The normalized relevance scores.
        """
        return self.__retrieve("linear_opposite", lambda: self.DEFAULT_MAX - self.linear())


    def squared(self) -> torch.Tensor:
        """
        Squared normalization of the relevance scores.
        The normalization is done by normalizing the relevance scores and then squaring them.

        :return: The normalized relevance scores.
        """
        return self.__retrieve("squared", lambda: self.linear() ** 2)
    

    def squared_opposite(self) -> torch.Tensor:
        """
        Squared normalization of the relevance scores, with opposite values.
        The normalization is done by normalizing the relevance scores and then squaring them,
        with the opposite values.

        :return: The normalized relevance scores.
        """
        return self.__retrieve("squared_opposite", lambda: self.DEFAULT_MAX - self.squared())
    

    def sigmoid(self, threshold: float = DEFAULT_THRESHOLD, amplitude: float = DEFAULT_AMPLITUDE) -> torch.Tensor:
        """
        Sigmoid normalization of the relevance scores; the output is in the range [0; 1].
        The normalization is done by applying the sigmoid function to the linearized relevance scores in [0; 1].
        More specifically, the linearized scores are multiplied by the `amplitude` factor and then passed through the sigmoid function,
        such that the scores above the `threshold` value become > 0.5 and the scores below become < 0.5.

        :return: The normalized relevance scores.
        """
        return self.__retrieve(f"sigmoid_{threshold}_{amplitude}", lambda: torch.sigmoid((self.linear() - threshold) * amplitude))
    

    def sigmoid_opposite(self, threshold: float = DEFAULT_THRESHOLD, amplitude: float = DEFAULT_AMPLITUDE) -> torch.Tensor:
        """
        Sigmoid normalization of the relevance scores, with opposite values.
        The normalization is done by applying the sigmoid function to the linearized relevance scores in [0; 1],
        with the opposite values.

        :return: The normalized relevance scores.
        """
        return self.__retrieve(f"sigmoid_opposite_{threshold}_{amplitude}", lambda: self.DEFAULT_MAX - self.sigmoid(threshold, amplitude))
    

    def sigmoid_adaptive(self, amplitude: float = DEFAULT_AMPLITUDE) -> torch.Tensor:
        """
        Sigmoid normalization of the relevance scores; the output is in the range [0; 1].
        The normalization is done by applying the sigmoid function to the linearized relevance scores in [0; 1].
        More specifically, the linearized scores are multiplied by the `amplitude` factor and then passed through the sigmoid function,
        such that the scores above the average value are transformed into > 0.5, whereas the scores below are transformed into < 0.5.

        :return: The normalized relevance scores.
        """
        def normalization() -> torch.Tensor:
            average = torch.mean(self.linear())
            return self.sigmoid(average, amplitude)
        return self.__retrieve(f"sigmoid_adaptive_{amplitude}", normalization)