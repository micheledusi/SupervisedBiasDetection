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
	"""
	TEMPLATES_SELECTED_NUMBER = "templates_selected_number", DEFAULT_TEMPLATES_SELECTED_NUMBER, "TM"
	AVERAGE_TEMPLATES = "average_templates", DEFAULT_AVERAGE_TEMPLATES
	AVERAGE_TOKENS = "average_tokens", DEFAULT_AVERAGE_TOKENS
	DISCARD_LONGER_WORDS = "discard_longer_words", DEFAULT_DISCARD_LONGER_WORDS
	MAX_TOKENS_NUMBER = "max_tokens_number", DEFAULT_MAX_TOKENS_NUMBER, "TK"
	CLASSIFIER_TYPE = "classifier", DEFAULT_CLASSIFIER_TYPE, "CL"
	CROSSING_STRATEGY = "crossing_strategy", DEFAULT_CROSSING_STRATEGY, "CR"
	POLARIZATION_STRATEGY = "polarization_strategy", DEFAULT_POLARIZATION_STRATEGY, "PL"

	def __new__(cls, str_value: str, default: Any, abbr: str = None):
		obj = object.__new__(cls)
		obj._value_: str = str_value
		obj._default_: Any = default
		obj._abbr_: str = abbr
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


class Configurations:
	"""
	This class represents a set of configurations, used in the program.
	It is a dictionary-like object, where the keys are the parameters and the values are the corresponding values.
	It is possible to access the values of the configurations using the square brackets notation. If the key is not
	contained in the set of configurations, a default value can be provided. If no default value is provided, the default
	value of the parameter is used. This default value comes from the "const" module.
	"""
    
	def __init__(self, values: dict[Parameter, Any] = None) -> None:
		"""
		Initializes the Configurations object.
		"""
		if not values:
			self.__configs: dict[Parameter, Any] = {}
		else:
			self.__configs = values.copy()

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
	
	def set(self, key: Parameter, value: Any) -> None:
		"""
		Adds a parameter value to the set of configurations.

		:param key: The key.
		:param value: The value of the configuration.
		"""
		self.__configs[key] = value
	
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
	
	def get_configurations_as_string(self) -> str:
		"""
		Gets the set of configurations as a string.

		:return: The set of configurations as a string.
		"""
		return "\n".join([f"\t> {str(k.value):30s}: {str(v)}" for k, v in self.__configs.items()])
	
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
		return self.get_configurations_as_string()
	
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

		:param keys: The keys.
		:return: The set of configurations as a string.
		"""
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
		self.__indices: dict[Parameter, int] = {}

	def __iter__(self) -> "ConfigurationsGrid":
		# Reset indices to 0s
		for key in self.__values.keys():
			if isinstance(self.__values[key], list):
				self.__indices[key] = 0
			else:
				continue
		return self
	
	def __next__(self) -> Configurations:
		configs = Configurations()
		for key in self.__values.keys():
			if isinstance(self.__values[key], list):
				configs.set(key, self.__values[key][self.__indices[key]])
			else:
				configs.set(key, self.__values[key])
		self.__increment_indices()
		if self.__has_ended():
			raise StopIteration
		return configs
	
	def __increment_indices(self) -> None:
		for key in self.__values.keys():
			if isinstance(self.__values[key], list):
				self.__indices[key] += 1
				if self.__indices[key] >= len(self.__values[key]):
					self.__indices[key] = 0
			else:
				continue
	
	def __has_ended(self) -> bool:
		for key in self.__values.keys():
			if isinstance(self.__values[key], list):
				if self.__indices[key] != 0:
					return False
			else:
				continue
		return True