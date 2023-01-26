# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#							   	#
#   Author:  Michele Dusi	   	#
#   Date:	2023			   	#
# - - - - - - - - - - - - - - - #

import torch
from datasets import Dataset

from data_processing.sentence_maker import get_dataset_from_words_csv
from model.embedding.word_embedder import WordEmbedder
from utility.cache import CachedData
from utility.const import DEVICE

DEFAULT_TEMPLATES_SELECTED_NUMBER = 'all'
DEFAULT_AVERAGE_TEMPLATES = True
DEFAULT_AVERAGE_TOKENS = True
DEFAULT_DISCARD_LONGER_WORDS = True
DEFAULT_MAX_TOKENS_NUMBER = 1


def get_cached_embeddings(property_name: str, property_pattern: str, words_file: str, templates_file: str, **kwargs) -> Dataset:
	"""
	Creates and returns a dataset with the embeddings of the words in the given file.
	The embeddings are cached, so that they are not computed again if the cache is not expired.

	:param property_name: the name of the property
	:param property_pattern: the pattern of the words that are replaced in the templates
	:param words_file: the path to the file containing the words
	:param templates_file: the path to the file containing the templates
	:param kwargs: the parameters for the WordEmbedder
	"""
	# Parameters
	params = {
		'templates_selected_number': kwargs.get('templates_selected_number', DEFAULT_TEMPLATES_SELECTED_NUMBER),
		'average_templates': kwargs.get('average_templates', DEFAULT_AVERAGE_TEMPLATES),
		'average_tokens': kwargs.get('average_tokens', DEFAULT_AVERAGE_TOKENS),
		'discard_longer_words': kwargs.get('discard_longer_words', DEFAULT_DISCARD_LONGER_WORDS),
		'max_tokens_number': kwargs.get('max_tokens_number', DEFAULT_MAX_TOKENS_NUMBER),
	}

	def create_embedding_fn() -> Dataset:
		# Loading the datasets
		templates: Dataset = Dataset.from_csv(templates_file)
		words: Dataset = get_dataset_from_words_csv(words_file)
		# Creating the word embedder
		word_embedder = WordEmbedder(pattern=property_pattern, **params)
		# Embedding words
		embedding_dataset = word_embedder.embed(words, templates)
		embedding_dataset = embedding_dataset.with_format('torch', device=DEVICE)
		return embedding_dataset

	# Creating info for the cache
	name: str = property_name + '_words'
	group: str = 'embedding'
	metadata: dict = params.copy()
	metadata.update({
		'property': property_name,
		'input_words_file': words_file,
		'input_templates_file': templates_file,
	})

	return CachedData(name, group, metadata, creation_fn=create_embedding_fn).__enter__()
	