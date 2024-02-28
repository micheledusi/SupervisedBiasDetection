# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi	 	#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module offers some methods to manage a "session"	of the program, i.e. a set of parameters and the corresponding values.

from enum import Enum
import logging
from typing import Any, Union
from colorist import BgColor, Color

from deprecated import deprecated

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

	BIAS_TEST = "bias_test", DEFAULT_BIAS_TEST, "BiTS"
	""" The statistical test to use to detect the bias on the contingency table.
	If `chi2`, the chi-squared test is used to detect the bias. """ # Note: this is the only strategy available at the moment.

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

	It is possible to access and set the values of each parameter using the square brackets notation. 
	The `get` method also accepts a default value, which is used in case the key is not present in the configuration. 
	If no default value is provided nor the key is recognized, each `Parameter` has its own "base" value.
	Using the "base" value will cause a warning in the logging system.

	Moreover, each Parameter can be associated with: (1) a single value, or (2) with a list of values.
	In the latter case, it is possible to iterate over all the possible combinations of the values of a selected set of parameters,
	through the `Configurations.Iterator` class, which can be obtained using the `iterate_over` method.
	"""

	ROW_PARAMETER_TEMPLATE: str = f"\t> {Color.BLUE}%-33s{Color.OFF}: {BgColor.BLACK}%s{BgColor.OFF} (%s) %s"
    
	def __init__(self, values: dict[Parameter, Any] = None, mutables: set[Parameter] = {}, use_defaults: bool = True) -> None:
		"""
		Initializes the Configurations object.

		:param values: The values of the parameters (optional). If not provided, the inner set of configurations is empty, but the default values of the parameters are used.
		:param mutables: The mutable parameters (optional), i.e. the parameters that are changing. If not provided, the list of mutables is empty.
		:param use_defaults: Whether to use the default values of the parameters, if no values are provided (optional). If True, the default values of the parameters are used.
		If false, a ValueError is raised if no values are provided.
		"""
		self.__dict: dict[Parameter, Any]
		""" The values of the parameters. It contains both atomic values and lists of values. """
		self.__mutables: frozenset[Parameter]
		""" The mutable parameters, i.e. the parameters that are changing thanks to the effect of some iterator. 
		Note: the "mutable" parameters are not the parameters associated with a list of values, but the parameters that are changing.
		The mutable parameters are used only to keep track of the current configuration."""
		self.__use_base_values: bool = use_defaults

		# In case no values are provided
		if not values:
			# We need to use the default values.
			# If the default values are required not to be used, we raise an error
			if not self.__use_base_values:
				raise ValueError("No values provided for the set of configurations in which default values are not admissible.")
			# Otherwise, we use the default values indeed
			self.__dict = {}
		# Otherwise, in case the values are provided
		else:
			self.__dict = values.copy()

		# If we cannot access the default values, we need to check that all the mutables are contained in the set of configurations
		if not self.__use_base_values:
			# We check that all the mutables are contained in the set of configurations
			for mutable in mutables:
				if mutable not in self.__dict:
					raise ValueError(f"The mutable parameter '{mutable}' is not contained in the provided set of configurations.")
		# If we can access the default values, we can add the mutables to the set of configurations
		if not mutables:
			self.__mutables = frozenset()
		else:
			self.__mutables = frozenset(mutables)


	@property
	def keys(self) -> tuple[Parameter]:
		"""
		Gets the keys of the parameters.

		:return: The keys of the parameters, as a tuple.
		"""
		return tuple(self.__dict.keys())
	

	@property
	def mutables(self) -> frozenset[Parameter]:
		"""
		Gets the mutable parameters.
		Please, consider that this method returns a copy of the mutable parameters, not the original set;
		thus, the original set cannot be modified, and furthermore the memory usage is doubled.

		:return: The mutable parameters, as a frozenset.
		"""
		return self.__mutables
	

	@property
	def use_base_values(self) -> bool:
		"""
		Checks if the default values of the parameters are used.

		:return: True if the default values of the parameters are used, False otherwise.
		"""
		return self.__use_base_values


	def __contains__(self, key: Parameter) -> bool:
		"""
		Checks if a parameter is contained in the set of configurations.

		:param key: The key.
		:return: True if the parameter is contained in the set of configurations, False otherwise.
		"""
		return key in self.__dict


	def __getitem__(self, key: Parameter) -> Any:
		"""
		Gets the value of a parameter.

		:param key: The key.
		:return: The value of the configuration.
		"""
		return self.get(key)
	

	@deprecated
	def __setitem__(self, key: Parameter, value: Any) -> None:
		"""
		Adds a parameter value to the set of configurations.

		:param key: The key.
		:param value: The value of the configuration.
		"""
		logging.error("The method '__setitem__' is deprecated. This class is meant to be used as a static and fixed collection.")
		self.set(key, value)


	@deprecated
	def __delitem__(self, key: Parameter) -> None:
		"""
		Deletes a parameter from the set of configurations.

		:param key: The key.
		"""
		logging.error("The method '__delitem__' is deprecated. This class is meant to be used as a static and fixed collection.")
		if key not in self.__dict:
			raise KeyError(f"The key '{key}' is not contained in the set of configurations.")
		del self.__dict[key]
	

	@deprecated
	def set(self, key: Parameter, value: Any) -> None:
		"""
		Adds a parameter value to the set of configurations.

		This method is deprecated: it is recommended to use the `Configurations` object as a static and fixed collection.

		:param key: The key.
		:param value: The value of the configuration.
		"""
		logging.error("The method '__delitem__' is deprecated. This class is meant to be used as a static and fixed collection.")
		self.__dict[key] = value
	

	def get(self, key: Parameter, default_value: Any = None) -> Any:
		"""
		Gets the value of a parameter.
		It also accepts a default value, which is used in case the key is not present in the configuration.
		However, if no particular default value is provided and the key is not recognized, we have two possibilities:
		- If the default values can be used, the "base" value of the parameter will be used. This will cause a warning in the logging system.
		- If the default values cannot be used, a ValueError is raised.

		:param key: The key.
		:param default_value: The default value (optional).
		:return: The value of the configuration.
		"""
		if key not in self.__dict:
			# If the key is not recognized, we return the default value or the "base" value of the parameter
			# If there's no default value
			if not default_value:
				# We check if the default 'base' values can be used
				if self.__use_base_values:
					logging.warning(f"The key '{key}' is not contained in the set of configurations. The 'base' value of the parameter will be used.")
					return key.default
				else:
					raise ValueError(f"The key '{key}' is not contained in the set of configurations and the 'base' values cannot be used.")
			# If there's a default value, we use it
			else:
				return default_value
		# If the key is recognized, we return the value
		else:	
			return self.__dict[key]


	def subget(self, *keys: list[Parameter]) -> "Configurations":
		"""
		Copy the current `Configurations` object to another object with only the selected parameters.
		In other words, it gets a "subset" of the configurations.

		All the other settings remain the same: if a parameter is mutable, it will be mutable in the new object as well.
		The usage of the "base" values is the same as in the original object.

		:param keys: The keys defining the subset of configurations.
		:return: The "subset" of the configurations.
		"""
		subdict: dict[Parameter, Any] = {key: self.get(key) for key in keys}
		submutables: list[Parameter] = [key for key in keys if key in self.__mutables]
		return Configurations(subdict, submutables, self.__use_base_values)


	def subget_mutables(self) -> "Configurations":
		"""
		Gets another Configurations object with only the mutable parameters.

		:return: The "subset" of the configurations.
		"""
		return self.subget(*self.__mutables)


	def subget_immutables(self) -> "Configurations":
		"""
		Gets another Configurations object with only the immutable parameters.

		:return: The "subset" of the configurations.
		"""
		immutables: list[Parameter] = [key for key in self.__dict.keys() if key not in self.__mutables]
		return self.subget(*immutables)


	def to_strdict(self) -> dict[str, Any]:
		"""
		Retrieves the set of configurations as a dictionary,
		but with the keys represented as strings.

		:return: The set of configurations as a dictionary.
		"""
		return {k.value: v for k, v in self.__dict.items()}


	def get_configurations_as_string(self) -> str:
		"""
		Gets the set of configurations as a string.

		:return: The set of configurations as a string.
		"""
		def get_value_string(key: Parameter) -> str:
			if key in self.__mutables:
				return self.ROW_PARAMETER_TEMPLATE % (key.value, str(self.get(key)), type(self.get(key)).__name__, "[mutable]")
			else:
				return self.ROW_PARAMETER_TEMPLATE % (key.value, str(self.get(key)), type(self.get(key)).__name__, "")
		return "\n".join([get_value_string(key) for key in self.__dict])	
	

	def to_abbrstr(self, *keys: list[Parameter]) -> str:
		"""
		Gets the set of configurations as a string, using the abbreviations of the parameters.
		The parameters are separated by an underscore. The order of the parameters is the same as the order of the keys.
		Each parameter is represented by its abbreviation followed by its value, with no spaces.

		:param keys: The keys (optional). If not provided, all the parameters are used.
		:return: The set of configurations as a string.
		"""
		if not keys:
			keys = self.__dict.keys()
		vals: list[str] = [f"{key.abbr}{self.get(key)}" for key in keys]
		return "_".join(vals)


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
		return f"{Color.BLUE}{self.to_abbrstr()}{Color.OFF}"


	def __eq__(self, __value: object) -> bool:
		"""
		Checks if two Configurations objects are equal.
		Two configurations are equal if and only if they have the same set of parameters and the same values for each parameter.
		Having the "same values" means that the values are equal, or that the values are lists with the same elements in the same order.
		"""
		if not isinstance(__value, Configurations):
			return False
		return self.__dict == __value.__dict
	

	def __ne__(self, __value: object) -> bool:
		"""
		Checks if two Configurations objects are different.
		Two configurations are different if they have different set of parameters or different values for at least one parameter.
		"""
		return not self.__eq__(__value)
	

	def __hash__(self) -> int:
		"""
		Computes the hash of the Configurations object.
		"""
		return hash(frozenset(self.__dict.items()))
	

	def iterate_over(self, keys: Union[list[Parameter], "Configurations.ParametersSelection"]) -> "Configurations.Iterator":
		"""
		Gets an iterator over all the possible combinations of the values of the parameters.

		:return: The iterator.
		"""
		changing_parameters: tuple[Parameter] 
		if isinstance(keys, list):
			changing_parameters = tuple(keys)
		elif isinstance(keys, Configurations.ParametersSelection):
			changing_parameters = keys.value
		else:
			raise ValueError("The keys must be either a list of parameters or a Configurations.ParametersSelection.")
		return Configurations.Iterator(self, changing_parameters)
	

	class Iterator:
		"""
		This class represents an iterator over all the possible combinations of configurations.

		It is used to iterate over all the possible combinations of the values of the parameters (i.e. the cartesian product);
		thus, it may take a long time to complete.

		Each combination of values is represented by a Configurations object.
		"""

		def __init__(self, original_configs: "Configurations", changing_parameters: tuple[Parameter]) -> None:
			self.__ref_configs: "Configurations" = original_configs
			""" The reference configurations. """
			self.__changing_parameters: tuple[Parameter] = tuple(key for key in changing_parameters if isinstance(self.original[key], list))
			""" The changing parameters, only if they have multiple values (i.e. the associated value is a list). """
			self.__indices: dict[Parameter, int] = {key: 0 for key in self.__changing_parameters}
			""" The indices of the current combination of values, only for the changing parameters. """
			self.__has_just_started: bool = False

			logging.info(f"Initialized the iterator over the configurations with the following changing parameters: {changing_parameters}")


		@property
		def original(self) -> "Configurations":
			"""
			Gets the original configurations.

			:return: The original configurations.
			"""
			return self.__ref_configs


		def __iter__(self) -> "Configurations.Iterator":
			"""
			Initializes the iterator.
			It returns the iterator itself, and it is called implicitly at the beginning of the iteration.
			For this reason, it's used to reset the iterator.
			"""
			# We setup the flag for the first iteration
			self.__has_just_started = True
			return self
		

		def __next__(self) -> "Configurations":
			# [0] We check if the iteration has ended
			if self.__has_ended():
				raise StopIteration
			else:
				self.__has_just_started = False

			# [1] We compute the current combination of values based on the indices
			def get_current_value(key: Parameter) -> Any:
				if key in self.__changing_parameters:
					return self.original[key][self.__indices[key]]
				else:
					return self.original[key]
			orig_keys: tuple[Parameter] = self.original.keys
			curr_dict: dict[Parameter, Any] = {key: get_current_value(key) for key in orig_keys}
			curr_mutables: frozenset[Parameter] = self.original.mutables.union(self.__changing_parameters)
			configs: Configurations = Configurations(curr_dict, curr_mutables, self.original.use_base_values)

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
				if self.__indices[curr_key] >= len(self.original[curr_key]):
					self.__indices[curr_key] = 0
					curr_index += 1
				else:
					break
		

		def __has_ended(self) -> bool:
			"""
			Checks if the iteration has ended, by checking if all the indices are at the end of the list of values.
			"""
			return not self.__has_just_started and sum([i for i in self.__indices.values()]) == 0
		
	
	class ParametersSelection(Enum):
		"""
		This class represents the possible selections of parameters for the computation of the raw embeddings.
		"""
		RAW_EMBEDDINGS_COMPUTATION = (
			Parameter.MODEL_NAME, 
			Parameter.MAX_TOKENS_NUMBER, 
			Parameter.LONGER_WORD_POLICY,
		)
		""" The parameters used to configure the raw embeddings computation. """
		EMBEDDINGS_COMBINATION = (
			Parameter.WORDS_SAMPLING_PERCENTAGE, 
			Parameter.TEMPLATES_PER_WORD_SAMPLING_PERCENTAGE, 
			Parameter.TEMPLATES_POLICY, 
			Parameter.MAX_TESTCASE_NUMBER, 
			Parameter.CENTER_EMBEDDINGS,
		)
		""" The parameters used to configure the embeddings combination in testcases. """
		EMBEDDINGS_REDUCTION = (
			Parameter.REDUCTION_CLASSIFIER_TYPE,
			Parameter.EMBEDDINGS_DISTANCE_STRATEGY,
		)
		""" The parameters used to configure the reduction of the embeddings. """
		BIAS_EVALUTATION = (
			Parameter.CROSS_CLASSIFIER_TYPE,
			Parameter.BIAS_TEST,
		)
		""" The parameters used to configure the bias evaluation. """

		def __new__(cls, *values: tuple[Parameter]):
			obj = object.__new__(cls)
			obj._value_: tuple[Parameter] = values 	# type: ignore
			return obj
		
		@property
		def value(self) -> tuple[Parameter]:
			"""
			Gets the value of the selection.

			:return: The value of the selection.
			"""
			return self._value_


class Configurable:
	"""
	This class represents an object that can be configured using a set of configurations.
	A `Configured` is initialized with a set of configurations and a list of `Parameter` objects.
	The parameters are used to filter the configurations values: only the values of the configurations that correspond
	to the parameters are used to configure the object. The other values are discarded.
	"""

	def __init__(self, configs: Configurations, parameters: list[Parameter]) -> None:
		if not configs:
			raise ValueError("The set of configurations cannot be empty.")
		if not parameters or len(parameters) == 0:
			raise ValueError("The list of parameters cannot be empty. At least one parameter must be provided.")
		self.__configs: Configurations = configs.subget(*parameters)


	@property
	def configs(self) -> Configurations:
		"""
		Gets the configurations of the object.

		This property is read-only and provides a useful handle to the configurations of the object,
		which can be used in the subclasses to configure the object.

		:return: The configurations of the object.
		"""
		return self.__configs