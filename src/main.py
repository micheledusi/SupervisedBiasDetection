# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module contains the launcher of the project

import torch
from datasets import disable_caching as dataset_disable_caching
from datasets.utils import logging as datasets_logging
import logging
from experiments.dynamic_midstep import DynamicPipelineExperiment

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Torch setup
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)    # For reproducibility
# Datasets setup
dataset_disable_caching()
datasets_logging.set_verbosity_error()
datasets_logging.disable_progress_bar()

from data_processing.data_reference import PropertyDataReference
from utils.config import Configurations, Parameter
from utils.const import MODEL_NAME_BERT_BASE_UNCASED, MODEL_NAME_ROBERTA_BASE, MODEL_NAME_DISTILBERT_BASE_UNCASED


PROTECTED_PROPERTY = PropertyDataReference("gender", 1, 1)
# STEREOTYPED_PROPERTY = PropertyDataReference("gender", 1, 1)

# STEREOTYPED_PROPERTY = PropertyDataReference("profession", 4, 1)

# PROTECTED_PROPERTY = PropertyDataReference("religion", 1, 1)
# STEREOTYPED_PROPERTY = PropertyDataReference("religion", 1, 1)

# PROTECTED_PROPERTY = PropertyDataReference("quality", 1, 1)
# STEREOTYPED_PROPERTY = PropertyDataReference("quality", 1, 1)

# PROTECTED_PROPERTY = PropertyDataReference("ethnicity", 1, 1)
# STEREOTYPED_PROPERTY = PropertyDataReference("criminality", 1, 1)

#---> Controls
# PROTECTED_PROPERTY = PropertyDataReference("dogsandcats", 1, 1)
STEREOTYPED_PROPERTY = PropertyDataReference("dogsandcats", 1, 1)


configs = Configurations({
	# Raw embeddings computation
	Parameter.MODEL_NAME: MODEL_NAME_BERT_BASE_UNCASED,
	Parameter.MAX_TOKENS_NUMBER: 'all',
	Parameter.LONGER_WORD_POLICY: 'truncate',
	# Combining embeddings in single testcases
	Parameter.WORDS_SAMPLING_PERCENTAGE: 1.0,
	Parameter.TEMPLATES_PER_WORD_SAMPLING_PERCENTAGE: 1.0,
	Parameter.TEMPLATES_POLICY: 'average',
	Parameter.MAX_TESTCASE_NUMBER: 1,
	# Testcase post-processing
	Parameter.CENTER_EMBEDDINGS: False,
	# Reduction
	Parameter.REDUCTION_CLASSIFIER_TYPE: 'svm',
	Parameter.EMBEDDINGS_DISTANCE_STRATEGY: 'euclidean',
	# Bias evaluation
	Parameter.CROSS_CLASSIFIER_TYPE: 'svm',
	Parameter.BIAS_TEST : 'chi2',
})


if __name__ == "__main__":
	exp: DynamicPipelineExperiment = DynamicPipelineExperiment(configs)
	exp.run(prot_prop=PROTECTED_PROPERTY, ster_prop=STEREOTYPED_PROPERTY)
