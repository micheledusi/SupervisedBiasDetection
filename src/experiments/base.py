# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# Base class for the experiments.


from abc import abstractmethod
import time
from datasets import Dataset

from utils.caching.creation import PropertyDataReference, get_cached_embeddings
from utils.config import Configurations

REBUILD = False


class Experiment:
	"""
	The base class for the experiments.
	"""
	def __init__(self, name: str):
		"""
		The initializer for the experiment class.

		:param name: The name of the experiment.
		"""
		self._name = name
	
	@property
	def name(self) -> str:
		"""
		The name of the experiment.

		:return: The name of the experiment.
		"""
		return self._name
	
	def run(self, **kwargs) -> None:
		"""
		Runs the experiment.
		"""
		start_time = time.time()
		self._execute(**kwargs)
		end_time = time.time()
		print(f"Experiment {self.name} completed in {end_time - start_time} seconds.")
	
	@staticmethod
	def _get_property_embeddings(property: PropertyDataReference, configs: Configurations) -> Dataset:
		"""
		Returns the embeddings for the given property.

		:param kwargs: Additional arguments to pass to the embedding function.
		:return: The embeddings for the given property.
		"""
		# Extracting data from property object
		# 	property: The name of the property.
		# 	property_type: The type of the property. Must be "protected" or "stereotyped".
		# 	words_file_id: The id of the words file to use. (e.g. 01, 02, etc.)
		# 	templates_file_id: The id of the templates file to use. (e.g. 01, 02, etc.)

		# Retrieving embeddings dataset from cache
		embeddings: Dataset = get_cached_embeddings(property, configs=configs, rebuild=REBUILD)
		squeezed_embs = embeddings['embedding'].squeeze().tolist()
		embeddings = embeddings.remove_columns('embedding').add_column('embedding', squeezed_embs).with_format('torch')
		return embeddings
	
	@staticmethod
	def _get_embeddings(protected_property: PropertyDataReference, stereotyped_property: PropertyDataReference, configs: Configurations) -> tuple[Dataset, Dataset]:
		"""
		Returns the embeddings for the protected and stereotyped property.

		:param protected_property: The reference to the protected property data (name, type, words_file_id, templates_file_id).
		:param stereotyped_property: The reference to the stereotyped property data (name, type, words_file_id, templates_file_id).
		:param num_max_tokens: The number of maximum tokens to consider in the embeddings.
		:param num_templates: The number of templates to consider in the embeddings.
		:return: A tuple containing the embeddings for the protected and stereotyped property.
		"""
		protected_embedding_dataset = Experiment._get_property_embeddings(protected_property, configs).sort('word')
		stereotyped_embedding_dataset = Experiment._get_property_embeddings(stereotyped_property, configs).sort('word')
		if 'descriptor' in stereotyped_embedding_dataset.column_names:
			stereotyped_embedding_dataset = stereotyped_embedding_dataset.filter(lambda x: x['descriptor'] != 'unused')
		return protected_embedding_dataset, stereotyped_embedding_dataset

	@abstractmethod
	def _execute(self, **kwargs) -> None:
		"""
		Description and execution of the core experiment.
		"""
		raise NotImplementedError