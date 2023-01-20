# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# Base class for the experiments.


from abc import abstractmethod
import time


class Experiment:

    def __init__(self, name: str):
        """
        The initializer for the experiment class.

        :param name: The name of the experiment.
        """
        self._name = name
    
    @property
    def name(self) -> str:
        """
        The name of the experiment.

        :return: The name of the experiment.
        """
        return self._name
    
    def run(self, **kwargs) -> None:
        """
        Runs the experiment.
        """
        start_time = time.time()
        self._execute(**kwargs)
        end_time = time.time()
        print(f"Experiment {self.name} completed in {end_time - start_time} seconds.")
    
    @abstractmethod
    def _execute(self, **kwargs) -> None:
        """
        Description and execution of the core experiment.
        """
        raise NotImplementedError