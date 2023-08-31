# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module contains the launcher of the project.

import torch
from datasets import disable_caching
from data_processing.data_reference import PropertyDataReference

from experiments.classification import ClassificationExperiment
from experiments.dim_reduction import DimensionalityReductionExperiment
from experiments.dim_reduction_comparison import DimensionalityReductionsComparisonExperiment
from experiments.midstep_analysis_2 import MidstepAnalysis2Experiment
from experiments.midstep_chi_squared import MidstepAnalysisChiSquared
from experiments.separation import SeparationExperiment

# Libraries setup
torch.manual_seed(42)
disable_caching()

PROTECTED_PROPERTY = PropertyDataReference("religion", 2, 1)
STEREOTYPED_PROPERTY = PropertyDataReference("quality", 1, 1)
MIDSTEP: int = 42

if __name__ == "__main__":

	# 
	### Regarding one single property	
	#
	# ClassificationExperiment().run(prot_prop=PROTECTED_PROPERTY)
	# SeparationExperiment().run(prot_prop=PROTECTED_PROPERTY, midstep=MIDSTEP)
	
	#
	### Regarding two properties and their relationship, with a specific midstep
	#
	# DimensionalityReductionExperiment().run(prot_prop=PROTECTED_PROPERTY, ster_prop=STEREOTYPED_PROPERTY, midstep=MIDSTEP)
	# DimensionalityReductionsComparisonExperiment().run(prot_prop=PROTECTED_PROPERTY, ster_prop=STEREOTYPED_PROPERTY, midstep=MIDSTEP)

	#
	### Regarding two properties and their relationship, for all midsteps
	#
	MidstepAnalysis2Experiment().run(prot_prop=PROTECTED_PROPERTY, ster_prop=STEREOTYPED_PROPERTY)
	# MidstepAnalysisChiSquared().run(prot_prop=PROTECTED_PROPERTY, ster_prop=STEREOTYPED_PROPERTY)
