# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module contains the launcher of the project

import torch
from datasets import disable_caching, logging
from data_processing.data_reference import PropertyDataReference

from experiments.midstep_chi_squared import MidstepAnalysisChiSquared
from utils.config import ConfigurationsGrid, Parameter

# Libraries setup
torch.manual_seed(42)
disable_caching()
logging.set_verbosity_error()

PROTECTED_PROPERTY = PropertyDataReference("religion", 1, 1)
STEREOTYPED_PROPERTY = PropertyDataReference("quality", 1, 1)
MIDSTEP: int = 62

configurations = ConfigurationsGrid({
	Parameter.MAX_TOKENS_NUMBER: 'all',
	Parameter.TEMPLATES_SELECTED_NUMBER: 'all',
	Parameter.CLASSIFIER_TYPE: 'svm',
	Parameter.REDUCTION_TYPE: 'pca',
	Parameter.CENTER_EMBEDDINGS: False,
})

if __name__ == "__main__":
	# For every combination of parameters, run the experiment
	for configs in configurations:
		experiment = MidstepAnalysisChiSquared(configs)
		experiment.run(prot_prop=PROTECTED_PROPERTY, ster_prop=STEREOTYPED_PROPERTY)
	