# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi	 	#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module offers some utils functions for caching data.

import copy
from datetime import datetime
import json
import os
from typing import Any, Callable
import pickle as pkl
import warnings
from datasets import Dataset

from model.binary_scoring.base import BinaryScorer
from model.binary_scoring.crossing.base import CrossingScorer
from model.binary_scoring.crossing.factory import CrossingFactory
from model.binary_scoring.polarization.base import PolarizationScorer
from model.binary_scoring.polarization.factory import PolarizationFactory
from utils.config import Configurations, Parameter
from utils.const import *
from data_processing.sentence_maker import get_dataset_from_words_csv, get_generation_datasets
from model.embedding.word_embedder import WordEmbedder


def _hash_dict(obj: Any) -> int:
	"""
	Makes a hash from a dictionary, list, tuple or set to any level, that contains
	only other hashable types (including any lists, tuples, sets, and
	dictionaries).
	"""
	if isinstance(obj, (set, tuple, list)):
		return tuple([_hash_dict(elem) for elem in obj])	

	elif not isinstance(obj, dict):
		return hash(obj)

	new_o = copy.deepcopy(obj)
	for k, v in new_o.items():
		new_o[k] = _hash_dict(v)

	return hash(tuple(frozenset(sorted(new_o.items()))))


class CacheManager:
	"""
	This class offers some utils functions for caching data.
	Each data can be saved along with a unique identifier and some metadata.
	Then, the data can be retrieved by providing the unique identifier and the metadata.
	"""
	DEFAULT_ROOT_DIR: str = 'cache'
	DEFAULT_GROUP: str = 'generic'

	REGISTER_FILENAME: str = 'register.json'

	def __init__(self, root_dir: str = DEFAULT_ROOT_DIR) -> None:
		self.root: str = root_dir

	def _init_group(self, group: str) -> None:
		"""
		This method initializes a new group of data, i.e. the directory where the data will be saved.
		If the group already exists, this method does nothing.
		If the directory where the data will be saved does not exist, this method creates it.
		Plus, this method creates a file containing the register of the group, called 'register.json'.

		:param group: The name of the group.
		:return: None.
		"""
		# If the group directory does not exist, create it.
		if not os.path.exists(self.root + '/' + group):
			os.makedirs(self.root + '/' + group)
		# If the register file does not exist, create it.
		if not os.path.exists(self.root + '/' + group + '/' + self.REGISTER_FILENAME):
			with open(self.root + '/' + group + '/' + self.REGISTER_FILENAME, 'w') as f:
				f.write('[]')

	def _compute_filename(self, identifier: str, group: str, metadata: dict[str, Any]) -> str:
		"""
		This method computes the filename of the object with the given identifier, group and metadata.
		The filename is generated by hashing the metadata, the identifier and the group.
		"""
		descriptor = metadata.copy()
		descriptor['identifier'] = identifier
		descriptor['group'] = group
		return str(_hash_dict(descriptor))

	def _register(self, identifier: str, filename: str, group: str, metadata: dict[str, Any]) -> None:
		"""
		This method register a new object in the correct group.
		The registration writes in the group register file (a JSON file) the identifier and the metadata of the object.
		Default metadata are added to the given metadata, such as:
		- the save timestamp
		
		This method does not save the object itself.
		This method does not check if the object already exists.
		This method does not check if the group exists: it is the caller's responsibility to check it and to create the group if necessary.

		:param identifier: The unique identifier of the object.
		:param filename: The filename of the object, generated by the hash of the metadata (see method ``save``)
		:param group: The name of the group
		:param metadata: The metadata of the object, as a dictionary
		:return: None.
		"""
		register = json.load(open(self.root + '/' + group + '/' + self.REGISTER_FILENAME, 'r'))
		assert isinstance(register, list)

		# If corresponding objects already exist, we remove them
		def is_corresponding(entry):
			return entry['identifier'] == identifier and entry['metadata'] == metadata
		annotated_register = map(lambda entry: (entry, is_corresponding(entry)), register)
		register = []
		for entry, to_be_deleted in annotated_register:
			if not to_be_deleted:
				register.append(entry)
			else:
				print("Removing old object from cache: ", entry['filename'])
				os.remove(self.root + '/' + group + '/' + entry['filename'] + '.pkl')
		# "register" has now been cleaned from the corresponding objects (if any)
		# Also, the corresponding files have been deleted from the cache.

		# Adding the new object to the register
		register.append({
			'identifier': identifier,
			'filename': filename,
			'timestamp': str(datetime.today()),
			'metadata': metadata
		})
		# Saving the register
		json.dump(register, open(self.root + '/' + group + '/' + self.REGISTER_FILENAME, 'w'), indent=4)

	def save(self, data: Any, identifier: str, group: str | None = None, metadata: dict[str, Any] | None = None) -> None:
		"""
		This method saves the given data along with the given identifier and metadata.
		The data is saved in a pickle file, and the metadata is saved in the group register file.

		:param data: The data to save.
		:param identifier: The unique identifier of the data.
		:param group: The name of the group (optional; default: None)
		:param metadata: The metadata of the object to save (optional; default: None)
		:return: None.
		"""
		# Asserting that the group exists
		if not group:
			group = self.DEFAULT_GROUP
		self._init_group(group)
		# Asserting the metadata are a dictionary
		if not metadata:
			metadata = dict()
		# Computing the filename
		filename = self._compute_filename(identifier, group, metadata)
		# Registering the object to the group register
		self._register(identifier, filename, group, metadata)
		# Saving the data
		pkl.dump(data, open(self.root + '/' + group + '/' + filename + '.pkl', 'wb'))

	def exists(self, identifier: str, group: str | None = None, metadata: dict[str, Any] | None = None) -> bool:
		"""
		This method checks if the object with the given identifier, group and metadata exists.
		If the object exists, this method returns True.
		If the object does not exist, this method returns False.

		:param identifier: The unique identifier of the object.
		:param group: The name of the group (optional; default: None)
		:param metadata: The metadata of the object, as a dictionary (optional; default: None)
		:return: True if the object exists, False otherwise.
		"""
		if not group:
			group = self.DEFAULT_GROUP
		if not metadata:
			metadata = dict()

		# If the group does not exist, the object does not exist
		if not os.path.exists(self.root + '/' + group):
			return False
		# If the register file does not exist, the object does not exist
		if not os.path.exists(self.root + '/' + group + '/' + self.REGISTER_FILENAME):
			return False
		# If the object is not in the register, the object does not exist
		register = json.load(open(self.root + '/' + group + '/' + self.REGISTER_FILENAME, 'r'))
		assert isinstance(register, list)
		for obj in register:
			if obj['identifier'] == identifier and obj['metadata'] == metadata:
				return True

	def _retrieve_filename(self, identifier: str, group: str, metadata: dict[str, Any]) -> str | None:
		"""
		This method returns the filename of the object with the given identifier and metadata.
		If the object does not exist, this method returns None.

		:param identifier: The unique identifier of the object.
		:param group: The name of the group (optional; default: None)
		:param metadata: The metadata of the object, as a dictionary (optional; default: None)
		:return: The filename of the object, or None if the object does not exist.
		"""
		# Checking if the object exists
		if not self.exists(identifier, group, metadata):
			return None
		# Searching
		register = json.load(open(self.root + '/' + group + '/' + self.REGISTER_FILENAME, 'r'))
		assert isinstance(register, list)
		for obj in register:
			if obj['identifier'] == identifier and obj['metadata'] == metadata:
				return obj['filename']
		return None
	
	def load(self, identifier: str, group: str | None = None, metadata: dict[str, Any] | None = None) -> Any:
		"""
		This method loads the object with the given identifier and metadata.
		If the object does not exist, this method raises an exception.

		:param identifier: The unique identifier of the object.
		:param group: The name of the group (optional; default: None)
		:param metadata: The metadata of the object, as a dictionary (optional; default: None)
		:return: The object.
		"""
		# Checking if the object exists
		if not group:
			group = self.DEFAULT_GROUP
		if not metadata:
			metadata = dict()
		# The check on the specific object is done in the `_retrieve_filename` method
		filename = self._retrieve_filename(identifier, group, metadata)
		if not filename:
			raise Exception("The object with identifier '" + identifier + "' and metadata '" + str(metadata) + "' does not exist in group '" + group + "'")

		# Loading the object
		return pkl.load(open(self.root + '/' + group + '/' + filename + '.pkl', 'rb'))


class CachedData:
	def __init__(self, name: str, group: str, metadata: dict[str, Any], creation_fn: Callable = None, rebuild: bool = False) -> None:
		"""
		This class is a context manager that can be used to cache data.
		If the data does not exist in the cache, it is created using the given creation function.
		If the data exists in the cache, it is loaded from the cache.

		:param name: The name of the object.
		:param group: The name of the group.
		:param metadata: The metadata of the object.
		:param creation_fn: The function to call to create the object if it does not exist in the cache (optional; default: None)
		:param rebuild: If True, the object is always created from scratch (optional; default: False)
		"""
		self._name: str = name
		self._group: str = group
		self._metadata: dict[str, Any] = metadata
		self._creation_fn: Callable = creation_fn
		self._rebuild: bool = rebuild
		self._cacher: CacheManager = CacheManager()

	def __enter__(self):
		if not self._rebuild and self._cacher.exists(self._name, self._group, self._metadata):
			try:
				print(f"Loading cached data \"{self._name}\"")
				return self._cacher.load(self._name, self._group, self._metadata)
			except FileNotFoundError as e:
				print(f"File not found for cached data \"{self._name}\": {e}")
		# Otherwise
		if self._creation_fn is None:
			raise NotImplemented("The creation function is not defined, and the object does not exist in the cache.")
		else:
			print(f"Creating data \"{self._name}\" from scratch and saving to the cache")
			data = self._creation_fn()
			self._cacher.save(data, self._name, self._group, self._metadata)
			return data
	
	def __exit__(self, *args):
		pass


def get_params_embeddings(configs: Configurations) -> dict[str, Any]:
	"""
	Returns the parameters used for embedding words.
	"""
	return configs.subget(
		Parameter.TEMPLATES_SELECTED_NUMBER,
		Parameter.AVERAGE_TEMPLATES,
		Parameter.AVERAGE_TOKENS,
		Parameter.DISCARD_LONGER_WORDS,
		Parameter.MAX_TOKENS_NUMBER,
	).to_strdict()


def get_params_cross_scores(configs: Configurations) -> dict[str, Any]:
	"""
	Returns the parameters used for the cross-scoring, i.e. the procedure that score each pair of protected and stereotyped words.
	"""
	return configs.subget(
		Parameter.DISCARD_LONGER_WORDS,
		Parameter.MAX_TOKENS_NUMBER,
		Parameter.CROSSING_STRATEGY,
	).to_strdict()


def get_params_polarization_scores(configs: Configurations) -> dict[str, Any]:
	"""
	Returns the parameters used for the polarization scoring, i.e. the procedure that combines the cross-scoring scores for pairs of protected values.
	"""
	return configs.subget(
		Parameter.DISCARD_LONGER_WORDS,
		Parameter.MAX_TOKENS_NUMBER,
		Parameter.CROSSING_STRATEGY,
		Parameter.POLARIZATION_STRATEGY,
	).to_strdict()


def get_cached_embeddings(property_name: str, property_pattern: str, words_file: str, templates_file: str, configs: Configurations, rebuild: bool = False) -> Dataset:
	"""
	Creates and returns a dataset with the embeddings of the words in the given file.
	The embeddings are cached, so that they are not computed again if the cache is not expired.

	:param property_name: the name of the property
	:param property_pattern: the pattern of the words that are replaced in the templates
	:param words_file: the path to the file containing the words
	:param templates_file: the path to the file containing the templates
	:param kwargs: the parameters for the WordEmbedder
	"""
	params = get_params_embeddings(configs)

	def create_embedding_fn() -> Dataset:
		# Disabling annoying "FutureWarning" messages
		warnings.simplefilter(action='ignore', category=FutureWarning)
		# Loading the datasets
		templates: Dataset = Dataset.from_csv(templates_file)
		words: Dataset = get_dataset_from_words_csv(words_file)
		# Creating the word embedder
		word_embedder = WordEmbedder(configs, pattern=property_pattern)
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

	return CachedData(name, group, metadata, creation_fn=create_embedding_fn, rebuild=rebuild).__enter__()


def get_cached_cross_scores(protected_property: str, stereotyped_property: str, generation_id: int, configs: Configurations, rebuild: bool = False) -> tuple[Dataset, Dataset, torch.Tensor]:
	params = get_params_cross_scores(configs)

	def create_cross_scores_fn() -> Dataset:
		pp_words, sp_words, templates = get_generation_datasets(protected_property, stereotyped_property, generation_id)
		scorer: CrossingScorer = CrossingFactory.create(configs)
		results = scorer.compute(templates, pp_words, sp_words)
		return results
	
	# Creating info for the cache
	name: str = f"{protected_property}_{stereotyped_property}_cross_scores"
	group: str = 'crossing_scores'
	metadata: dict = params.copy()
	metadata.update({
		'protected_property': protected_property,
		'stereotyped_property': stereotyped_property,
		'generation_id': generation_id,
	})

	return CachedData(name, group, metadata, creation_fn=create_cross_scores_fn, rebuild=rebuild).__enter__()


def get_cached_polarization_scores(protected_property: str, stereotyped_property: str, generation_id: int, configs: Configurations, rebuild: bool = False) -> tuple[tuple[str], tuple[str], Dataset]:
	params = get_params_polarization_scores(configs)

	def create_polarization_scores_fn() -> tuple[tuple[str], tuple[str], Dataset]:
		# Retrieving the cross scores from cache
		# The result is a tuple of two datasets and a tensor: (pp_values, sp_values, cross_scores)
		outcomes = get_cached_cross_scores(protected_property, stereotyped_property, generation_id, configs, rebuild=rebuild)
		middle_outcomes = BinaryScorer.prepare_crossing_scores_for_polarization(*outcomes)
		scorer: PolarizationScorer = PolarizationFactory.create(configs)
		return scorer(*middle_outcomes)
		
	# Creating info for the cache
	name: str = f"{protected_property}_{stereotyped_property}_polarization_scores"
	group: str = 'polarization_scores'
	metadata: dict = params.copy()
	metadata.update({
		'protected_property': protected_property,
		'stereotyped_property': stereotyped_property,
		'generation_id': generation_id,
	})

	return CachedData(name, group, metadata, creation_fn=create_polarization_scores_fn, rebuild=rebuild).__enter__()

