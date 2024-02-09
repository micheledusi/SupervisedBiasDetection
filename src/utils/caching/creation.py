# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi	 	#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module offers some utils functions for caching data.

import warnings
import torch
from datasets import Dataset

from data_processing.data_reference import BiasDataReference, PropertyDataReference
from data_processing.sentence_maker import get_dataset_from_words_csv, get_generation_datasets
from data_processing.types import CachedDataType
from model.binary_scoring.base import BinaryScorer
from model.binary_scoring.crossing.base import CrossingScorer
from model.binary_scoring.crossing.factory import CrossingFactory
from model.binary_scoring.polarization.base import PolarizationScorer
from model.binary_scoring.polarization.factory import PolarizationFactory
from utils.caching.manager import CachedData
from utils.config import Configurations
from utils.const import DEVICE
from model.embedding.word_embedder import RawEmbedder, WordEmbedder


def get_cached_embeddings(property: PropertyDataReference, configs: Configurations, rebuild: bool = False) -> Dataset:
	"""
	Creates and returns a dataset with the embeddings of the words in the given file.
	The embeddings are cached, so that they are not computed again if the cache is not expired.

	:param property: the property to use, a PropertyDataReference object with the following fields:
		- name: the name of the property
		- type: the type of the property. Must be "protected" or "stereotyped"
		- words_file_id: the id of the words file to use. (e.g. 01, 02, etc.)
		- templates_file_id: the id of the templates file to use. (e.g. 01, 02, etc.)
	:param kwargs: the parameters for the WordEmbedder
	"""
	params = CachedDataType.EMBEDDINGS.get_relevant_configs(configs).to_strdict()

	def create_embedding_fn() -> Dataset:
		# Disabling annoying "FutureWarning" messages
		warnings.simplefilter(action='ignore', category=FutureWarning)
		# Loading the datasets
		templates: Dataset = Dataset.from_csv(property.templates_file)
		words: Dataset = get_dataset_from_words_csv(property.words_file)
		# Creating the word embedder
		word_embedder = WordEmbedder(configs)
		# Embedding words
		embedding_dataset = word_embedder.embed(words, templates)
		embedding_dataset = embedding_dataset.with_format('torch', device=DEVICE)
		return embedding_dataset

	# Creating info for the cache
	metadata: dict = params.copy()
	metadata.update({
		'property': property.name,
		'input_words_file': property.words_file,
		'input_templates_file': property.templates_file,
	})

	return CachedData(CachedDataType.EMBEDDINGS, metadata, creation_fn=create_embedding_fn, rebuild=rebuild).__enter__()


def get_cached_raw_embeddings(property: PropertyDataReference, configs: Configurations, rebuild: bool = False) -> Dataset:
	"""
	Creates and returns a dataset with the embeddings in the "raw" format.
	The embeddings are cached, so that they are not computed again if the cache is not expired.

	:param property: the property to use, a PropertyDataReference object with the following fields:
		- name: the name of the property
		- type: the type of the property. Must be "protected" or "stereotyped"
		- words_file_id: the id of the words file to use. (e.g. 01, 02, etc.)
		- templates_file_id: the id of the templates file to use. (e.g. 01, 02, etc.)
	:param kwargs: the parameters for the WordEmbedder
	"""
	params = CachedDataType.RAW_EMBEDDINGS.get_relevant_configs(configs).to_strdict()

	def create_embedding_fn() -> Dataset:
		# Disabling annoying "FutureWarning" messages
		warnings.simplefilter(action='ignore', category=FutureWarning)
		# Loading the datasets
		templates: Dataset = Dataset.from_csv(property.templates_file)
		words: Dataset = get_dataset_from_words_csv(property.words_file)
		# Creating the word embedder
		word_embedder = RawEmbedder(configs)
		# Embedding words
		embedding_dataset = word_embedder.embed(words, templates)
		return embedding_dataset

	# Creating info for the cache
	metadata: dict = params.copy()
	metadata.update({
		'property': property.name,
		'input_words_file': property.words_file,
		'input_templates_file': property.templates_file,
	})

	return CachedData(CachedDataType.RAW_EMBEDDINGS, metadata, creation_fn=create_embedding_fn, rebuild=rebuild).__enter__()


def get_cached_crossing_scores(bias_reference: BiasDataReference, configs: Configurations, rebuild: bool = False) -> tuple[Dataset, Dataset, torch.Tensor]:
	params = CachedDataType.CROSSING_SCORES.get_relevant_configs(configs).to_strdict()

	def create_cross_scores_fn() -> tuple[Dataset, Dataset, torch.Tensor]:
		pp_words, sp_words, templates = get_generation_datasets(bias_reference)
		scorer: CrossingScorer = CrossingFactory.create(configs)
		results = scorer.compute(templates, pp_words, sp_words)
		return results
	
	# Creating info for the cache
	metadata: dict = params.copy()
	metadata.update({
		'protected_property': bias_reference.protected_property.name,
		'stereotyped_property': bias_reference.stereotyped_property.name,
		'generation_id': bias_reference.generation_id,
	})

	return CachedData(CachedDataType.CROSSING_SCORES, metadata, creation_fn=create_cross_scores_fn, rebuild=rebuild).__enter__()


def get_cached_polarization_scores(bias_reference: BiasDataReference, configs: Configurations, rebuild: bool = False) -> tuple[tuple[str], tuple[str], Dataset]:
	params = CachedDataType.POLARIZATION_SCORES.get_relevant_configs(configs).to_strdict()

	def create_polarization_scores_fn() -> tuple[tuple[str], tuple[str], Dataset]:
		# Retrieving the cross scores from cache
		# The result is a tuple of two datasets and a tensor: (pp_values, sp_values, polarization_scores)
		outcomes = get_cached_crossing_scores(bias_reference, configs, rebuild=rebuild)
		middle_outcomes = BinaryScorer.prepare_crossing_scores_for_polarization(*outcomes)
		scorer: PolarizationScorer = PolarizationFactory.create(configs)
		return scorer(*middle_outcomes)
		
	# Creating info for the cache
	metadata: dict = params.copy()
	metadata.update({
		'protected_property': bias_reference.protected_property.name,
		'stereotyped_property': bias_reference.stereotyped_property.name,
		'generation_id': bias_reference.generation_id,
	})

	return CachedData(CachedDataType.POLARIZATION_SCORES, metadata, creation_fn=create_polarization_scores_fn, rebuild=rebuild).__enter__()


