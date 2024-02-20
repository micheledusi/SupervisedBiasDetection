# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module contains the launcher of the project

import torch
from datasets import Dataset, disable_caching as dataset_disable_caching
from datasets.utils import logging as datasets_logging
import logging
from experiments.dynamic_midstep import DynamicPipelineExperiment
from model.embedding.center import EmbeddingCenterer

from model.embedding.combinator import EmbeddingsCombinator

# Logging setup
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Torch setup
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)    # For reproducibility
# Datasets setup
dataset_disable_caching()
datasets_logging.set_verbosity_error()
datasets_logging.disable_progress_bar()

from data_processing.data_reference import PropertyDataReference
from utils.caching.creation import get_cached_raw_embeddings
from utils.config import Configurations, ConfigurationsGrid, Parameter
from utils.const import MODEL_NAME_BERT_BASE_UNCASED, MODEL_NAME_ROBERTA_BASE, MODEL_NAME_DISTILBERT_BASE_UNCASED


REBUILD_DATASETS = False

# PROTECTED_PROPERTY = PropertyDataReference("gender", 1, 1)
# STEREOTYPED_PROPERTY = PropertyDataReference("profession", 3, 1)
PROTECTED_PROPERTY = PropertyDataReference("religion", 1, 1)
# STEREOTYPED_PROPERTY = PropertyDataReference("quality", 1, 1)
# PROTECTED_PROPERTY = PropertyDataReference("ethnicity", 1, 1)
STEREOTYPED_PROPERTY = PropertyDataReference("criminality", 1, 1)


# Raw embeddings computation
configurations_raw_embeddings = ConfigurationsGrid({
	Parameter.MODEL_NAME: [MODEL_NAME_BERT_BASE_UNCASED],
	Parameter.MAX_TOKENS_NUMBER: 'all',
	Parameter.LONGER_WORD_POLICY: 'truncate',
})
configurations_combined_embeddings = ConfigurationsGrid({
	# Combining embeddings in single testcases
	Parameter.WORDS_SAMPLING_PERCENTAGE: [0.9],
	Parameter.TEMPLATES_PER_WORD_SAMPLING_PERCENTAGE: [0.9],
	Parameter.TEMPLATES_POLICY: 'average',
	Parameter.MAX_TESTCASE_NUMBER: 20,
	# Testcase post-processing
	Parameter.CENTER_EMBEDDINGS: False,
})
configurations_bias_detection = Configurations({
	# Reduction
	Parameter.REDUCTION_CLASSIFIER_TYPE: 'svm',
	Parameter.EMBEDDINGS_DISTANCE_STRATEGY: 'euclidean',
	# Bias evaluation
	Parameter.CROSS_CLASSIFIER_TYPE: 'svm',
	Parameter.BIAS_TEST : 'chi2',
})


if __name__ == "__main__":
	# For every combination of parameters, run the experiment
	for configs_re in configurations_raw_embeddings:
		logging.debug("Configurations for the raw embeddings computation:\n%s", configs_re)
		
		# Loading the datasets
		protected_property_ds: Dataset = get_cached_raw_embeddings(PROTECTED_PROPERTY, configs_re, REBUILD_DATASETS)
		stereotyped_property_ds: Dataset = get_cached_raw_embeddings(STEREOTYPED_PROPERTY, configs_re, REBUILD_DATASETS)

		logging.debug(f"Resulting protected raw dataset for property \"{PROTECTED_PROPERTY.name}\":\n{protected_property_ds}")
		logging.debug(f"Resulting stereotyped raw dataset for property \"{STEREOTYPED_PROPERTY.name}\":\n{stereotyped_property_ds}")
	
		# Combining the embeddings
		combinator = EmbeddingsCombinator(configurations_combined_embeddings)

		combined_protected_embeddings: dict[Configurations, list[Dataset]] = combinator.combine(protected_property_ds)
		combined_stereotyped_embeddings: dict[Configurations, list[Dataset]] = combinator.combine(stereotyped_property_ds)

		# Centering the embeddings
		# For each configuration of the combined embeddings:
		for key_configs in combined_protected_embeddings:
			centerer: EmbeddingCenterer = EmbeddingCenterer(key_configs)
			# Centering the embeddings for each testcase
			combined_protected_embeddings[key_configs] = [centerer.center(ds) for ds in combined_protected_embeddings[key_configs]]
		for key_configs in combined_stereotyped_embeddings:
			centerer: EmbeddingCenterer = EmbeddingCenterer(key_configs)
			# Centering the embeddings for each testcase
			combined_stereotyped_embeddings[key_configs] = [centerer.center(ds) for ds in combined_stereotyped_embeddings[key_configs]]	

		# Now we have the combined embeddings, we can proceed with the bias detection
		logging.info("Configurations for the raw embeddings computation:\n%s", configs_re)
		for key_configs in combined_protected_embeddings:
			logging.info("Configurations for the combined embeddings:\n%s", key_configs)
			print(f"Configurations for the combined embeddings:\n{key_configs}")

			# We assume that the keys of the two dictionaries are the same
			# Meaning that the configurations are the same for both the protected and the stereotyped property
			protected_embs_ds_list: list[Dataset] = combined_protected_embeddings[key_configs]
			stereotyped_embs_ds_list: list[Dataset] = combined_stereotyped_embeddings[key_configs]

			experiment: DynamicPipelineExperiment = DynamicPipelineExperiment(configurations_bias_detection)
			experiment.run(
				prot_embs_ds_list=protected_embs_ds_list, 
				ster_embs_ds_list=stereotyped_embs_ds_list
				)
		