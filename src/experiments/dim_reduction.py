# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# EXPERIMENT: Dimensionality reduction over word embeddings obtained by BERT.
# DATE: 2023-01-20

import torch
from datasets import Dataset
from experiments.base import Experiment
from utility.cache import CachedData
from data_processing.sentence_maker import get_dataset_from_words_csv


class DimensionalityReductionExperiment(Experiment):

    def __init__(self) -> None:
        super().__init__("dimensionality reduction")

    def _execute(self, **kwargs) -> None:
        torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Parameters
        tested_property = 'profession'
        input_words_file = 'data/stereotyped-p/profession/words-01.csv'
        input_templates_file = 'data/stereotyped-p/profession/templates-01.csv'
        param_select_templates = 'all'
        param_average_templates = True
        param_average_tokens = True
        param_discard_longer_words = True
        param_max_tokens_number = 1
        
        # Disk management for embedding datasets
        name = 'profession_words'
        group = 'embedding'
        metadata = {
            'stereotyped_property': tested_property,
            'input_words': input_words_file,
            'input_templates': input_templates_file,
            'select_templates': param_select_templates,
            'average_templates': param_average_templates,
            'average_tokens': param_average_tokens,
            'discard_longer_words': param_discard_longer_words,
            'max_tokens_number': param_max_tokens_number,
        }

        def create_embedding_fn():
            from model.embedding.word_embedder import WordEmbedder
            # Loading the datasets
            templates: Dataset = Dataset.from_csv(input_templates_file)
            words: Dataset = get_dataset_from_words_csv(input_words_file)
            # Creating the word embedder
            word_embedder = WordEmbedder(select_templates=param_select_templates, average_templates=param_average_templates, average_tokens=param_average_tokens)
            # Embedding a word
            embedding_dataset = word_embedder.embed(words, templates)
            return embedding_dataset

        with CachedData(name, group, metadata, creation_fn=create_embedding_fn) as embedding_dataset:
            print(embedding_dataset)
