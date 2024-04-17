# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2024				#
# - - - - - - - - - - - - - - - #

# This module allows to plot a vector of weights.
# It can also shows multiple vectors at the same time.

import matplotlib.pyplot as plt
import torch

class WeightsPlotter:
    """
    This class is used to plot the weights.
    """

    def __init__(self, weights: dict[str, torch.Tensor] | torch.Tensor) -> None:
        """
        The initializer for the WeightsPlotter class.
        It can accept a dictionary of weights or a single tensor.
        In both case, the tensor is detached.

        :param weights: The weights to plot.
        """
        if isinstance(weights, torch.Tensor):
            self._weights = {"weights": weights.detach()}
        else:
            self._weights = {name: weights[name].detach() for name in weights}

    
    def show(self) -> "WeightsPlotter":
        """
        This method plots the weights.
        For each element in the "weights" dictionary, it plots the corresponding weights in a subplot.

        :return: The WeightsPlotter object.
        """
        fig, axs = plt.subplots(len(self._weights), 1, figsize=(7, 2 * len(self._weights)))
        if len(self._weights) == 1:
            axs: list = [axs]

        for i, (name, weights) in enumerate(self._weights.items()):
            axs[i].scatter(range(len(weights)), weights)
            axs[i].set_title(name)

        plt.show()
        return self