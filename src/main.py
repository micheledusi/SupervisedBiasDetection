# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module contains the launcher of the project.

from experiments.dim_reduction import DimensionalityReductionExperiment

if __name__ == "__main__":
    experiment = DimensionalityReductionExperiment()
    experiment.run()
