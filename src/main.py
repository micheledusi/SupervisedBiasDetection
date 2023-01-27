# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module contains the launcher of the project.

from experiments.dim_reduction import DimensionalityReductionExperiment
from experiments.midstep_analysis import MidstepAnalysisExperiment

if __name__ == "__main__":
	experiment = MidstepAnalysisExperiment()
	experiment.run()
