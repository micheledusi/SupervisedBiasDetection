# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module contains the launcher of the project.

import torch
from datasets import disable_caching

from experiments.dim_reduction import DimensionalityReductionExperiment
from experiments.midstep_analysis_2 import MidstepAnalysis2Experiment
from experiments.midstep_chi_squared import MidstepAnalysisChiSquared
from experiments.classification import ClassificationExperiment

# Libraries setup
torch.manual_seed(42)
# disable_caching()

if __name__ == "__main__":
	
	# MidstepAnalysisChiSquared().run()
	
	# MidstepAnalysis2Experiment().run()

	# DimensionalityReductionExperiment().run()

	ClassificationExperiment().run()
