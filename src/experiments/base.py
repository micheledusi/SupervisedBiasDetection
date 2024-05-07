# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# Base class for the experiments.


from abc import abstractmethod
import logging
import os
import time
from typing import Any
from datasets import Dataset, concatenate_datasets

from utils.caching.creation import PropertyDataReference, get_cached_embeddings, get_cached_raw_embeddings
from utils.config import Configurations
from utils import file_system as fs

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

REBUILD = False


class Experiment:
	"""
	The base class for the experiments.
	"""

	PROTECTED_PROPERTY_KEY: str = "prot_prop"
	STEREOTYPED_PROPERTY_KEY: str = "ster_prop"
	PROTECTED_EMBEDDINGS_DATASETS_LIST_KEY: str = "prot_embs_ds_list"
	STEREOTYPED_EMBEDDINGS_DATASETS_LIST_KEY: str = "ster_embs_ds_list"
	MIDSTEP_KEY: str = "midstep"


	def __init__(self, name: str, required_kwargs: list[str] = [], configs: Configurations = None):
		"""
		The initializer for the experiment class.

		:param name: The name of the experiment.
		:param required_kwargs: The list of required arguments to pass to the run method. If not specified, the run method will accept any argument.
		If specified, the run method will require the given arguments to work properly.
		:param configs: The configurations to use for the experiment.
		"""
		self._name = name
		self.__required_kwargs = required_kwargs
		self.__configs = configs

		# Check if the given configurations are valid
		if self.__configs is not None and not isinstance(self.__configs, Configurations):
			raise ValueError("The given configurations are not valid. Please provide a valid instance of Configurations.")
		
		# Setup contestual properties
		# These properties are used by the subclasses to perform the experiment, and are available only during the execution.
		# They're extracted from the arguments passed to the run method, and cancelled at the end of the execution.
		self._prot_prop: PropertyDataReference = None
		self._ster_prop: PropertyDataReference = None
		self._midstep: int = None

		# Flags
		self._is_executing: bool = False

		# Collector
		self.__results_collector: ResultCollector = ResultCollector(self)
	
	@property
	def name(self) -> str:
		"""
		The name of the experiment.

		:return: The name of the experiment.
		"""
		return self._name
	
	@property
	def configs(self) -> Configurations:
		"""
		The configurations used for the experiment.

		:return: The configurations to use for the experiment, as instance of class `Configurations`.
		"""
		return self.__configs
	
	@property
	def results_collector(self) -> "ResultCollector":
		"""
		The result collector for the experiment.

		:return: The result collector for the experiment.
		"""
		return self.__results_collector

	@property
	def protected_property(self) -> PropertyDataReference:
		"""
		The protected property.
		Note: this property is available only during the execution of the experiment.
		It is extracted from the arguments passed to the run method, and cancelled at the end of the execution.
		Calls outside the execution will raise an exception.

		:return: The protected property.
		"""
		if not self._is_executing:
			raise RuntimeError("The protected property is available only during the execution of the experiment.")
		return self._prot_prop
	
	@protected_property.setter
	def protected_property(self, value: PropertyDataReference) -> None:
		"""
		Sets the protected property.

		:param value: The protected property.
		"""
		if not self._is_executing:
			raise RuntimeError("The protected property is available only during the execution of the experiment.")
		self._prot_prop = value
	
	@property
	def stereotyped_property(self) -> PropertyDataReference:
		"""
		The stereotyped property.

		:return: The stereotyped property.
		"""
		if not self._is_executing:
			raise RuntimeError("The stereotyped property is available only during the execution of the experiment.")
		return self._ster_prop
	
	@stereotyped_property.setter
	def stereotyped_property(self, value: PropertyDataReference) -> None:
		"""
		Sets the stereotyped property.

		:param value: The stereotyped property.
		"""
		if not self._is_executing:
			raise RuntimeError("The stereotyped property is available only during the execution of the experiment.")
		self._ster_prop = value
	
	@property
	def midstep(self) -> int:
		"""
		The midstep.

		:return: The midstep.
		"""
		if not self._is_executing:
			raise RuntimeError("The midstep value is available only during the execution of the experiment.")
		return self._midstep
	
	@midstep.setter
	def midstep(self, value: int) -> None:
		"""
		Sets the midstep.

		:param value: The midstep.
		"""
		if not self._is_executing:
			raise RuntimeError("The midstep value is available only during the execution of the experiment.")
		self._midstep = value
	
	def _extract_value_from_kwargs(self, key: str, **kwargs) -> any:
		"""
		Extracts the value of the given key from the given arguments.

		:param key: The key of the value to extract.
		:param kwargs: The arguments passed to the experiment.
		:return: The value of the given key, or the default value if not present.
		"""
		# If the key is not required by the current experiment
		if key not in self.__required_kwargs:
			# But it's specified in the kwargs
			if key in kwargs:
				raise KeyError(f"The \"{key}\" field is not required for this experiment, but it was specified in the kwargs. Please check consistency.")
			# Otherwise, we simply return None to indicate that the value is not present
			return None

		# If the key is required by the current experiment
		# We check if it's present in the kwargs
		if key in kwargs:
			return kwargs[key]
		else:
			raise KeyError(f"The \"{key}\" field is required for this experiment, but it was not specified in the kwargs. Please provide a consistent value.")
	
	def _extract_protected_property(self, **kwargs) -> PropertyDataReference:
		"""
		Extracts the protected property from the given arguments.

		:param kwargs: The arguments passed to the experiment.
		:return: The protected property.
		"""
		return self._extract_value_from_kwargs(self.PROTECTED_PROPERTY_KEY, **kwargs)
	
	def _extract_stereotyped_property(self, **kwargs) -> PropertyDataReference:
		"""
		Extracts the stereotyped property from the given arguments.

		:param kwargs: The arguments passed to the experiment.
		:return: The stereotyped property.
		"""
		return self._extract_value_from_kwargs(self.STEREOTYPED_PROPERTY_KEY, **kwargs)
	
	def _extract_midstep(self, **kwargs) -> int:
		"""
		Extracts the midstep from the given arguments.

		:param kwargs: The arguments passed to the experiment.
		:return: The midstep.
		"""
		return self._extract_value_from_kwargs(self.MIDSTEP_KEY, **kwargs)
	
	def _get_results_folder(self, configs: Configurations, prot_values: list | Dataset = None, ster_values: list | Dataset = None) -> str:
		"""
		Returns the folder in which to save the results, based on the protected and stereotyped property.
		The first property is required, whereas the second is optional. If not specified, the folder will
		contain only the name of the protected property.
		
		Note: the returned string does NOT contain the final slash.
		Note: if the folder does not exist, it will be created.

		:param configs: The current configurations dictionary to use.
		:return: The folder in which to save the results.
		"""
		properties_strings: list[str] = []

		if self.protected_property is not None:
			if prot_values is None:
				try:
					prot_values = self._get_property_embeddings(self.protected_property, configs)['word']
					prot_str: str = self.protected_property.name_with_classes_number(prot_values)
				except:
					TypeError
					prot_str: str = self.protected_property.name
			properties_strings.append(prot_str)

		if self.stereotyped_property is not None:
			if ster_values is None:
				try:
					ster_values = self._get_property_embeddings(self.stereotyped_property, configs)['word']
					ster_str: str = self.stereotyped_property.name_with_classes_number(ster_values)
				except:
					TypeError
					ster_str: str = self.stereotyped_property.name
			properties_strings.append(ster_str)

		folder: str = f"{fs.get_model_results_folder(configs)}/{'-'.join(properties_strings)}"

		if not os.path.exists(folder):
			os.makedirs(folder)
		return folder

	def run(self, **kwargs) -> None:
		"""
		Runs the experiment.
		It can accepts additional arguments, such as:
			- prot_prop: The protected property to use.
			- ster_prop: The stereotyped property to use.
			- midstep: The midstep to use.
		
		Within the execution, these arguments will be available as instance properties with the names of
		protected_property, stereotyped_property and midstep. They will be cancelled at the end of the execution.
		If not specified, the default values will be used.
		"""
		# Setting up the experiment
		self._is_executing = True
		start_time = time.time()
		# Extracting properties from kwargs
		self.protected_property = self._extract_protected_property(**kwargs)
		self.stereotyped_property = self._extract_stereotyped_property(**kwargs)
		self.midstep = self._extract_midstep(**kwargs)
		# Executing the experiment
		self._execute(**kwargs)
		end_time = time.time()
		# Saving the results
		results_ds: Dataset = self.results_collector.get_results()
		if results_ds is not None and len(results_ds) > 0:
			results_folder: str = self._get_results_folder(self.configs)
			results_ds.to_csv(f"{results_folder}/{self.name}_{time.strftime('%Y%m%d-%H%M%S')}.csv", index=False)
		# Cleaning up
		self.protected_property = None
		self.stereotyped_property = None
		self.midstep = None
		self._is_executing = False
		logging.info(f"Experiment {self.name} completed in {end_time - start_time} seconds.")
	
	@staticmethod
	def _get_property_embeddings(property: PropertyDataReference, configs: Configurations) -> Dataset:
		"""
		Returns the embeddings for the given property.

		:param property: The reference to the property data (name, type, words_file_id, templates_file_id).
		:param configs: The configurations to use for the experiment.
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
	
	def _get_embeddings(self, configs: Configurations) -> tuple[Dataset, Dataset]:
		"""
		Returns the embeddings for the protected and stereotyped property.

		:param protected_property: The reference to the protected property data (name, type, words_file_id, templates_file_id).
		:param stereotyped_property: The reference to the stereotyped property data (name, type, words_file_id, templates_file_id).
		:return: A tuple containing the embeddings for the protected and stereotyped property.
		"""
		protected_embedding_dataset = Experiment._get_property_embeddings(self.protected_property, configs).sort('word')
		stereotyped_embedding_dataset = Experiment._get_property_embeddings(self.stereotyped_property, configs).sort('word')
		return protected_embedding_dataset, stereotyped_embedding_dataset

	@staticmethod
	def _get_property_raw_embeddings(property: PropertyDataReference, configs: Configurations, rebuild: bool=REBUILD) -> Dataset:
		"""
		Returns the embeddings for the given property, in the "raw" format.
		This means that the templates are not averaged, and the words are not combined in testcases.

		:param property: The reference to the property data (name, type, words_file_id, templates_file_id).
		:param configs: The configurations to use for the experiment.
		:return: The "raw" embeddings for the given property.
		"""
		# Retrieving embeddings dataset from cache
		embeddings: Dataset = get_cached_raw_embeddings(property, configs, rebuild)
		# squeezed_embs = embeddings['embedding'].squeeze().tolist()
		# embeddings = embeddings.remove_columns('embedding').add_column('embedding', squeezed_embs).with_format('torch')
		return embeddings
	
	def _get_raw_embeddings(self, configs: Configurations, rebuild: bool=REBUILD) -> tuple[Dataset, Dataset]:
		"""
		Returns the embeddings for the protected and stereotyped property.

		:param protected_property: The reference to the protected property data (name, type, words_file_id, templates_file_id).
		:param stereotyped_property: The reference to the stereotyped property data (name, type, words_file_id, templates_file_id).
		:return: A tuple containing the embeddings for the protected and stereotyped property.
		"""
		protected_embedding_dataset = Experiment._get_property_raw_embeddings(self.protected_property, configs, rebuild).sort(['template', 'word'])
		stereotyped_embedding_dataset = Experiment._get_property_raw_embeddings(self.stereotyped_property, configs, rebuild).sort(['template', 'word'])
		return protected_embedding_dataset, stereotyped_embedding_dataset

	@abstractmethod
	def _execute(self, **kwargs) -> None:
		"""
		Description and execution of the core experiment.
		"""
		raise NotImplementedError
	

class ResultCollector:
	"""
	A class to collect the results of the experiments.
	"""

	EXPERIMENT_NAME_COL: str = "experiment_name"

	def __init__(self, experiment: Experiment) -> None:
		"""
		The initializer for the result collector.

		:param experiment: The experiment to collect the results from.
		"""
		self.__experiment = experiment
		self.__results: Dataset = None

	
	def collect(self, current_configs: Configurations, results: dict[str, Any], remove_list_parameters: bool=False) -> None:
		"""
		Collects the results of the experiment.

		:param current_configs: The current configurations of the experiment with which the results were obtained.
		:param results: The results of the experiment.
		:param remove_list_parameters: If True, the configurations with list values will be removed from the results.
		"""
		# We analyse the structure of the results to understand how to collect them
		are_results_list: bool = True
		results_len: int = 1
		for key in results.keys():
			if not isinstance(results[key], list):
				are_results_list = False
				break
			else:
				results_len: int = len(results[key])
		# We verify that all the results have the same length
		if are_results_list:
			for key in results.keys():
				if len(results[key]) != results_len:
					raise ValueError("The results have different lengths. Please check consistency.")

		# We create a dictionary to collect the results
		current_results_dict: dict[str, list] = {}
		# We add a column for the experiment name
		current_results_dict[self.EXPERIMENT_NAME_COL] = [self.__experiment.name] * results_len
		# Now we add the configurations to the dictionary, such that each config_key is associated to a list
		for key in current_configs.keys:
			config_value = current_configs[key]
			# If the value is a list, we remove it if required
			if remove_list_parameters and isinstance(current_configs[key], list) or isinstance(current_configs[key], tuple):
				config_value = None
			current_results_dict[key.value] = [config_value] * results_len
		# Ad then we add the results to the dictionary
		for key in results.keys():
			if are_results_list:
				current_results_dict[key] = results[key]
			else:
				current_results_dict[key] = [results[key]]

		# Finally, we convert the results into a dataset
		results_ds: Dataset = Dataset.from_dict(current_results_dict)
		# We add the results to the internal results dataset
		if self.__results:
			try:
				self.__results = concatenate_datasets([self.__results, results_ds])
			except:
				ValueError
				logger.error("An error occurred while concatenating the results datasets. The features cannot be aligned:\n" + 
				f"\t\t> The original results have the following {len(self.__results.column_names)} columns: " + str(self.__results.column_names) + "\n" +
				f"\t\t> The new results have the following {len(results_ds.column_names)} columns: " + str(results_ds.column_names))
		else:
			self.__results = results_ds
	

	def get_results(self) -> Dataset:
		"""
		Returns the collected results.

		:return: The collected results.
		"""
		return self.__results