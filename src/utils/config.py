# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi	 	#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module offers some methods to manage a "session"	of the program, i.e. a set of parameters and the corresponding values.

from enum import Enum
from typing import Any

from utils.const import *


class Parameter(Enum):
	"""
	This class represents a parameter.
	A parameter is a variable that can be set to a value, and it is used to configure a run of the program.
	Each parameter has a name, a default value, and an abbreviation (optional, used to represent the parameter in a string).
	"""

	# Raw embeddings computation

	MODEL_NAME = "model_name", DEFAULT_MODEL_NAME, "MODEL"
	""" The name of the Large Language Model from the Huggingface library. """
	
	MAX_TOKENS_NUMBER = "max_tokens_number", DEFAULT_MAX_TOKENS_NUMBER, "TOK"
	""" The maximum number of tokens to consider in a word. The value can be an integer or `all` to consider all the tokens. """
	
	LONGER_WORD_POLICY = "longer_word_policy", DEFAULT_LONGER_WORD_POLICY, "LWP"
	""" How to manage words with more tokens than the maximum number of tokens.
	If `discard`, words with more tokens than the maximum number of tokens are discarded. 
	If `truncate`, the tokens are truncated to the maximum number of tokens. """

	AVERAGE_TOKENS = "average_tokens", True, "AVT"
	"""	Whether to average the tokens of a word. I true, all the tokens obtained from the tokenization of a word are averaged.
	@deprecated: this should be always set to True."""

	# Embeddings combination in testcases

	WORDS_SAMPLING_PERCENTAGE = "words_sampling_number", DEFAULT_WORDS_SAMPLING_NUMBER, "WS"
	""" The number of words to sample from the dataset. If `all`, all the words are used.
	If the number of words in the dataset is less than the number of words to sample, all the words are used. """

	TEMPLATES_PER_WORD_SAMPLING_PERCENTAGE = "templates_sampling_number", DEFAULT_TEMPLATES_SAMPLING_NUMBER, "TS"
	""" The number of templates to sample from the dataset. If `all`, all the templates are used. """
	
	TEMPLATES_POLICY = "templates_policy", DEFAULT_TEMPLATES_POLICY, "TP"
	""" The policy to use to combine the embeddings of the templates. 
	If `average`, all the embeddings from the same word but different embeddings are averaged in one embedding.
	If `distinct`, the embeddings of the same word are taken separately. """

	MAX_TESTCASE_NUMBER = "testcase_number", DEFAULT_MAX_TESTCASE_NUMBER, "TESC"
	""" The number of testcases to generate, for each combination of parameters.
	If the number of possible combinations is less than the number of testcases to generate, all the possible combinations are used. 
	(E.g. in subsampling 3 words among 4, the maximum number of possible combinations is 4)."""
	
	# Testcase post-processing

	CENTER_EMBEDDINGS = "center_embeddings", DEFAULT_CENTER_EMBEDDINGS, "CE"
	""" Whether to center the embeddings of the testcases. If true, the embeddings are centered, i.e. the mean of the embeddings is subtracted from each embedding. """

	# Reduction configurations

	REDUCTION_CLASSIFIER_TYPE = "reduction_classifier", DEFAULT_CLASSIFIER_TYPE, "ReCL"
	""" The classifier to use to reduce the embeddings. Possible values are `svm` and `linear`. """

	EMBEDDINGS_DISTANCE_STRATEGY = "embeddings_distance_strategy", DEFAULT_EMBEDDINGS_DISTANCE_STRATEGY, "EDIST"
	""" The strategy to use to compute the distance between the embeddings. Possible values are `euclidean` and `cosine`. """

	#### TODO: multiple parameters need to be added here for the reduction of the embeddings

	# Bias detection analysis on reduced embeddings

	CROSS_CLASSIFIER_TYPE = "cross_classifier", DEFAULT_CLASSIFIER_TYPE, "CrCL"
	""" The classifier to use to produce the contingency table, which crosses the protected and stereotyped words.
	Possible values are `svm` and `linear`. """

	BIAS_TEST = "bias_test", "chi2", "BiTS"
	""" The statistical test to use to detect the bias on the contingency table.
	If `chi_squared`, the chi-squared test is used to detect the bias. """ # Note: this is the only strategy available at the moment.

	# Other parameters
	TEST_SPLIT_PERCENTAGE = "test_split_percentage", DEFAULT_TEST_SPLIT_PERCENTAGE, "SP"
	CROSS_PROBABILITY_STRATEGY = "cross_probability_strategy", DEFAULT_CROSS_PROBABILITY_STRATEGY, "CR"
	POLARIZATION_STRATEGY = "polarization_strategy", DEFAULT_POLARIZATION_STRATEGY, "PL"
	REDUCTION_TYPE = "reduction", DEFAULT_REDUCTION_TYPE, "RD"

	def __new__(cls, str_value: str, default: Any, abbr: str = None):
		obj = object.__new__(cls)
		obj._value_: str = str_value 	# type: ignore	# For reasons that are unintelligible to humans, Python requires these comments to avoid a warning 
		obj._default_: Any = default 	# type: ignore
		obj._abbr_: str = abbr 			# type: ignore
		if abbr is not None:
			# Create alias
			cls._value2member_map_[abbr] = obj
		return obj
	
	@property
	def default(self) -> Any:
		"""
		Gets the default value of the parameter.

		:return: The default value of the parameter.
		"""
		return self._default_

	@property
	def abbr(self) -> str:
		"""
		Gets the abbreviation of the parameter.

		:return: The abbreviation of the parameter.
		"""
		return self._abbr_
	
	def __repr__(self) -> str:
		"""
		Represents the parameter with the name.
		"""
		return f"p\'{self._value_}\'"


class Configurations:
	"""
	This class represents a set of configurations, used in the program.
	It is a dictionary-like object, where the keys are the parameters and the values are the corresponding values.
	It is possible to access the values of the configurations using the square brackets notation. If the key is not
	contained in the set of configurations, a default value can be provided. If no default value is provided, the default
	value of the parameter is used. This default value comes from the "const" module.
	"""
    
	def __init__(self, values: dict[Parameter, Any] = None, mutables: list[Parameter] = []) -> None:
		"""
		Initializes the Configurations object.
		"""
		if not values:
			self.__configs: dict[Parameter, Any] = {}
			if mutables:
				raise ValueError("The list of mutables must be empty if no values are provided.")
			else:
				self.__mutables = mutables.copy()
		else:
			self.__configs = values.copy()
			self.__mutables = mutables.copy()

	def __contains__(self, key: Parameter) -> bool:
		"""
		Checks if a parameter is contained in the set of configurations.

		:param key: The key.
		:return: True if the parameter is contained in the set of configurations, False otherwise.
		"""
		return key in self.__configs
	
	def __getitem__(self, key: Parameter) -> Any:
		"""
		Gets the value of a parameter.

		:param key: The key.
		:return: The value of the configuration.
		"""
		return self.get(key)
	
	def __setitem__(self, key: Parameter, value: Any) -> None:
		"""
		Adds a parameter value to the set of configurations.

		:param key: The key.
		:param value: The value of the configuration.
		"""
		self.set(key, value)

	def __delitem__(self, key: Parameter) -> None:
		"""
		Deletes a parameter from the set of configurations.

		:param key: The key.
		"""
		if key not in self.__configs:
			raise KeyError(f"The key '{key}' is not contained in the set of configurations.")
		del self.__configs[key]
	
	def set(self, key: Parameter, value: Any, mutable: bool = False) -> None:
		"""
		Adds a parameter value to the set of configurations.

		:param key: The key.
		:param value: The value of the configuration.
		"""
		self.__configs[key] = value
		if mutable:
			self.__mutables.append(key)
		else:
			if key in self.__mutables:
				self.__mutables.remove(key)
	
	def get(self, key: Parameter, default_value: Any = None) -> Any:
		"""
		Gets the value of a parameter.

		:param key: The key.
		:return: The value of the configuration.
		"""
		if not default_value:
			# If no default value is provided, use the default value of the parameter
			return self.__configs.get(key, key.default)
		else:
			# Otherwise, use the provided default value
			return self.__configs.get(key, default_value)
	
	def subget(self, *keys: list[Parameter]) -> "Configurations":
		"""
		Gets another Configurations object with only the selected parameters.

		:param keys: The keys.
		:return: The "subset" of the configurations.
		"""
		configs = Configurations()
		for key in keys:
			configs.set(key, self.get(key))
		return configs
	
	def subget_mutables(self) -> "Configurations":
		"""
		Gets another Configurations object with only the mutable parameters.

		:return: The "subset" of the configurations.
		"""
		configs = Configurations()
		for key in self.__mutables:
			configs.set(key, self.get(key), mutable=True)
		return configs
	
	def subget_immutables(self) -> "Configurations":
		"""
		Gets another Configurations object with only the immutable parameters.

		:return: The "subset" of the configurations.
		"""
		configs = Configurations()
		for key in self.__configs:
			if key not in self.__mutables:
				configs.set(key, self.get(key))
		return configs
	
	def get_configurations_as_string(self) -> str:
		"""
		Gets the set of configurations as a string.

		:return: The set of configurations as a string.
		"""
		return "\n".join([f"\t> \033[36m{str(k.value):30s}\033[0m: \033[96m{str(v)}\033[0m" for k, v in self.__configs.items()])
	
	def __str__(self) -> str:
		"""
		Gets the set of configurations as a string.

		:return: The set of configurations as a string.
		"""
		return self.get_configurations_as_string()
	
	def __repr__(self) -> str:
		"""
		Gets the set of configurations as a string.

		:return: The set of configurations as a string.
		"""
		return f"\033[36m{self.to_abbrstr()}\033[0m"
	
	def to_strdict(self) -> dict[str, Any]:
		"""
		Gets the set of configurations as a dictionary.

		:return: The set of configurations as a dictionary.
		"""
		return {k.value: v for k, v in self.__configs.items()}
	
	def to_abbrstr(self, *keys: list[Parameter]) -> str:
		"""
		Gets the set of configurations as a string, using the abbreviations of the parameters.
		The parameters are separated by an underscore. The order of the parameters is the same as the order of the keys.
		Each parameter is represented by its abbreviation followed by its value, with no spaces.

		:param keys: The keys (optional). If not provided, all the parameters are used.
		:return: The set of configurations as a string.
		"""
		if not keys:
			keys = self.__configs.keys()
		vals: list[str] = [f"{key.abbr}{self.get(key)}" for key in keys]
		return "_".join(vals)


class ConfigurationsGrid:
	"""
	This class represents a grid of configurations, i.e. multiple configurations with different values for various parameters.
	It is also an iterator, so it can be used to iterate over all the configurations.
	The iteration method traverses all the possible combinations of the values of the parameters (i.e. the cartesian product);
	thus, it may take a long time to complete.

	Each combination of values is represented by a Configurations object.
	"""

	def __init__(self, values: dict[Parameter, list[Any] | Any]) -> None:
		self.__values: dict[Parameter, list[Any] | Any] = values
		""" The values of the parameters. Each value can be a single value or a list of values. """
		self.__indices: dict[Parameter, int] = {key: 0 for key in self.__values.keys() if isinstance(self.__values[key], list)}
		""" The indices of the current combination of values, only for the parameters with multiple values. """
		self.__changing_parameters: list[Parameter] = [key for key in self.__values.keys() if isinstance(self.__values[key], list)]
		""" The parameters with multiple values. """
		self.__has_just_started: bool = False

	def __iter__(self) -> "ConfigurationsGrid":
		"""
		Initializes the iterator.
		It returns the iterator itself, and it is called implicitly at the beginning of the iteration.
		For this reason, it's used to reset the iterator.
		"""
		# We setup the flag for the first iteration
		self.__has_just_started = True
		return self
	
	def __next__(self) -> Configurations:
		# [0] We check if the iteration has ended
		if self.__has_ended():
			raise StopIteration
		else:
			self.__has_just_started = False

		# [1] We compute the current combination of values based on the indices
		configs = Configurations()
		for key in self.__values.keys():
			if isinstance(self.__values[key], list):
				configs.set(key, self.__values[key][self.__indices[key]], mutable=True)
			else:
				configs.set(key, self.__values[key], mutable=False)

		# [2] We increment the indices
		self.__increment_indices()
		return configs
	
	def __increment_indices(self) -> None:
		"""	
		Increments the indices of the current combination of values.
		"""
		curr_index: int = 0
		while curr_index < len(self.__changing_parameters):
			curr_key: Parameter = self.__changing_parameters[curr_index]
			self.__indices[curr_key] += 1
			if self.__indices[curr_key] >= len(self.__values[curr_key]):
				self.__indices[curr_key] = 0
				curr_index += 1
			else:
				break
	
	def __has_ended(self) -> bool:
		"""
		Checks if the iteration has ended, by checking if all the indices are at the end of the list of values.
		"""
		return not self.__has_just_started and sum([i for i in self.__indices.values()]) == 0
	
	def subget(self, *keys: list[Parameter]) -> "ConfigurationsGrid":
		"""
		Gets another ConfigurationsGrid object with only the selected parameters.

		:param keys: The keys.
		:return: The "subset" of the configurations.
		"""
		tmp_dict:  dict[Parameter, list[Any] | Any] = {}
		for key in keys:
			tmp_dict[key] = self.__values[key]
		return ConfigurationsGrid(tmp_dict)


class Configurable:
	"""
	This class represents an object that can be configured using a set of configurations.
	A `Configured` is initialized with a set of configurations and a list of `Parameter` objects.
	The parameters are used to filter the configurations values: only the values of the configurations that correspond
	to the parameters are used to configure the object. The other values are discarded.
	"""

	def __init__(self, configs: Configurations | ConfigurationsGrid, parameters: list[Parameter]) -> None:
		if not configs:
			raise ValueError("The set of configurations cannot be empty.")
		if not parameters:
			raise ValueError("The list of parameters cannot be empty. At least one parameter must be provided.")
		self.__configs: Configurations | ConfigurationsGrid = configs.subget(*parameters)
		# FIXME: this is a temporary solution, for which both Configurations and ConfigurationsGrid are accepted.
		# In order to do so, both classes implement the same method "subget", which is used to filter the configurations.
		# This is not a good solution and should be changed in the future, with a better design allowing to use only one class.
	
	@property
	def configs(self) -> Configurations | ConfigurationsGrid:
		"""
		Gets the configurations of the object.

		:return: The configurations of the object.
		"""
		return self.__configs