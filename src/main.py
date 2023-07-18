# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module contains the launcher of the project.

import torch
from datasets import disable_caching

from experiments.classification import ClassificationExperiment
from experiments.dim_reduction import DimensionalityReductionExperiment
from experiments.dim_reduction_comparison import DimensionalityReductionsComparisonExperiment
from experiments.midstep_analysis_2 import MidstepAnalysis2Experiment
from experiments.midstep_chi_squared import MidstepAnalysisChiSquared
from experiments.separation import SeparationExperiment

# Libraries setup
torch.manual_seed(42)
# disable_caching()

if __name__ == "__main__":
	
	# ClassificationExperiment().run()
	
	DimensionalityReductionExperiment().run()

	# DimensionalityReductionsComparisonExperiment().run()

	# MidstepAnalysis2Experiment().run()

	# MidstepAnalysisChiSquared().run()

	# SeparationExperiment().run()
