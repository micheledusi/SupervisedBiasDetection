# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	 2024				#
# - - - - - - - - - - - - - - - #

# This module contains the class "EmbeddingsCombinator", which is responsible for combining the raw embeddings testcases.
# The "raw embeddings" are the embeddings of the single words and templates, computed by the "RawEmbedder" class.
# The combinator takes these embeddings and samples the appropriate number of words and templates, then combines them into one or multiple testcases.

import logging
import math
import random
from datasets import Dataset
import torch

from utils.config import Configurable, ConfigurationsGrid, Parameter


class EmbeddingsCombinator(Configurable):
    """
    Responsible for combining the raw embeddings testcases.
    The "raw embeddings" are the embeddings of the single words and templates, computed by the "RawEmbedder" class.
    The combinator takes these embeddings and samples the appropriate number of words and templates, then combines them into one or multiple testcases.
    """

    def __init__(self, configs: ConfigurationsGrid):
        """
        Initializes the combinator.
        :param raw_embedder: the raw embedder, which will provide the raw embeddings
        :param configurations: the configurations for the combinator
        """
        super(EmbeddingsCombinator, self).__init__(configs, parameters=[
            Parameter.WORDS_SAMPLING_PERCENTAGE,
            Parameter.TEMPLATES_PER_WORD_SAMPLING_PERCENTAGE,
            Parameter.TEMPLATES_POLICY,
            Parameter.MAX_TESTCASE_NUMBER,
			])
        
    def combine(self, raw_embeddings: Dataset) -> dict:
        """
        Combines the raw embeddings into testcases.
        The resulting object is a dictionary, where:
        - a key is a specific combination of configuration parameters.
        - a value is a list of Dataset objects, each representing a testcase.

        The Dataset objects contain the word embeddings on which the experimentation is done.
        Each Dataset is composed of the following fields:
        - "word": the string of the word
        - "value": the value that the word represents w.r.t. the protected/stereotyped property
        - "embedding": the embedding of the word, as a torch.Tensor

        :param raw_embeddings: the raw embeddings, as provided by the "RawEmbedder"
        :return: the combined embeddings
        """
        # Initializes the result
        combined_embeddings: dict = {}

        for config in self.configs:
            logging.debug("Current configuration for the combined embeddings computation:\n%s", config)

            # Gets the list of unique words
            unique_words: tuple[str] = tuple(set(raw_embeddings['word']))
            # Gets the number of rows to sample from the dataset
            words_to_sample = math.ceil(len(unique_words) * config[Parameter.WORDS_SAMPLING_PERCENTAGE])
            # Gets the number of unique values
            values_number = len(set(raw_embeddings['value']))

            # Sampling the dataset
            # [1] First, we sample the words
            testcases: list = []
            for _ in range(config[Parameter.MAX_TESTCASE_NUMBER]):

                # We must check if the dataset has enough values to be used in the experiment
                selected_values = 0
                while selected_values < values_number:

                    # Sampling random words from the unique list
                    sampled_words = random.sample(unique_words, k=words_to_sample)
                    # Then, we filter the dataset to keep only the rows with those words
                    first_selection: Dataset = raw_embeddings.filter(lambda x: x['word'] in sampled_words)
                    # Finally, we check if the first_selection has enough unique values
                    # If not, the while loop will repeat
                    selected_values = len(set(first_selection['value']))

                first_selection_length = len(first_selection)
                logging.debug(f"Selection by words: selected {first_selection_length} rows from the dataset, with {selected_values} unique values")

                # [2] Then, we sample the same first_selection, but w.r.t. the templates
                # Note: the templates depend on the words, thus we cannot sample them independently.
                # For this reason, we just discard a random number of rows, until we reach the desired number of rows
                indices = list(range(first_selection_length))
                rows_to_second_sample = math.ceil(first_selection_length * config[Parameter.TEMPLATES_PER_WORD_SAMPLING_PERCENTAGE])

                selected_values = 0
                while selected_values < values_number:

                    # Shuffling the indices
                    sampled_indices = random.sample(indices, k=rows_to_second_sample)
                    second_selection: Dataset = first_selection.select(sampled_indices)
                    # With this, we extract the unique values in the second_selection
                    selected_values = len(set(second_selection['value']))
                    
                logging.debug(f"Selection by templates: selected {len(second_selection)} rows from the first selecton, with {selected_values} unique values")

                # [3] Finally, if the policy is "average", we average the embeddings of the selected rows
                if config[Parameter.TEMPLATES_POLICY] == "average":
                    logging.info("Averaging the embeddings of the same word, for different templates")
                    # We copy the features of the current dataset selection, but as an empty dataset
                    final_selection: Dataset = second_selection.select([])
                    # For each word in the dataset
                    logging.debug("Sampled %d words", len(sampled_words))
                    for word in sampled_words:
                        word_selection: Dataset = second_selection.filter(lambda x: x['word'] == word)
                        assert len(word_selection) > 0, f"The word '{word}' is not sampled in the second-selection dataset"
                        final_selection = final_selection.add_item({
                            "word": word,
                            "value": word_selection['value'][0],
                            "descriptor": word_selection['descriptor'][0],
                            # "template": word_selection['template'], # We keep all the templates
                            # "sentence": word_selection['sentence'], # We keep all the sentences for the word
                            "embedding": torch.mean(word_selection['embedding'], dim=0).tolist(), # Averaging the embeddings on the first axis (=#templates)
                        })
                    logging.debug("Final selection size: %d", len(final_selection))
                elif config[Parameter.TEMPLATES_POLICY] == "distinct":
                    # Everything is already done
                    final_selection: Dataset = second_selection
                else:
                    # Unknown policy
                    raise ValueError(f"Unknown policy for the templates: {config[Parameter.TEMPLATES_POLICY]}")

                testcases.append(final_selection)

            combined_embeddings[config] = testcases
        
        return combined_embeddings

