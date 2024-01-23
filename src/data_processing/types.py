# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi	 	#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module defines the general "data" types that is used in the project.
# In particular, it defines:
# - A dataset of embeddings.
# - A dataset of crossing scores.
# - A dataset of polarization scores.

from enum import Enum
from typing import Type
from data_processing.json import CrossingScoresJSONConverter, EmbeddingsJSONConverter, PolarizationScoresJSONConverter

from utils.caching.stream import FileStream, JSONFileStream, PickleFileStream
from utils.config import Configurations, Parameter


class CachedDataType(Enum):
	"""
	This enum defines the possible types of data that can be cached.

	Each instance of the enum provides:
	- A value, which is the name of the type of data.
	- A filestream, which is the filestream to be used for caching the data.
	"""
	GENERIC = "generic", \
		(), \
		PickleFileStream
	EMBEDDINGS = "embeddings", \
		(
			Parameter.MODEL_NAME,
			Parameter.TEMPLATES_SELECTED_NUMBER,
			Parameter.AVERAGE_TEMPLATES,
			Parameter.AVERAGE_TOKENS,
			Parameter.DISCARD_LONGER_WORDS,
			Parameter.MAX_TOKENS_NUMBER,
		), \
		JSONFileStream, EmbeddingsJSONConverter
	CROSSING_SCORES = "crossing_scores", \
		(
			Parameter.MODEL_NAME,
			Parameter.DISCARD_LONGER_WORDS,
			Parameter.MAX_TOKENS_NUMBER,
			Parameter.CROSSING_STRATEGY,
		), \
		JSONFileStream, CrossingScoresJSONConverter
	POLARIZATION_SCORES = "polarization_scores", \
		(
			Parameter.MODEL_NAME,
			Parameter.DISCARD_LONGER_WORDS,
			Parameter.MAX_TOKENS_NUMBER,
			Parameter.CROSSING_STRATEGY,
			Parameter.POLARIZATION_STRATEGY,
		), \
		JSONFileStream, PolarizationScoresJSONConverter

	def __new__(cls, value: str, defining_parameters: list[Parameter] | tuple[Parameter], filestream_csl: Type[FileStream], *filestream_args):
		obj = object.__new__(cls)
		obj._value_ = value
		obj._defining_parameters_ = defining_parameters if isinstance(defining_parameters, tuple) else tuple(defining_parameters)
		if not filestream_csl:
			raise ValueError("The filestream for the type of data '{}' is not defined.".format(value))
		obj._filestream_cls_: Type[FileStream] = filestream_csl
		obj._filestream_args_ = filestream_args
		# The filestream class must offer a constructor with the following signature:
		# __init__(configs: Configurations, *args)
		obj._filestream = obj._filestream_cls_(*obj._filestream_args_)
		return obj
	
	@property
	def defining_parameters(self) -> tuple[Parameter]:
		"""
		Returns the parameters that define this type of data.
		More specifically, the parameters that define the type of data are the parameters that, if changed, will cause the data to be different.
		These are used to cache the data, so that the data is not recomputed if the parameters that define the data are not changed.

		:return: The parameters that define the type of data.
		"""
		return self._defining_parameters_
	
	@property
	def folder_name(self) -> str:
		"""
		Returns the name of the folder where the data of this type is cached.

		:return: The name of the folder where the data of this type is cached.
		"""
		return self.value
	
	def get_relevant_configs(self, configs: Configurations) -> Configurations:
		"""
		Returns a subset of the configurations, containing only the parameters that define this type of data.

		:param configs: The configurations.
		:return: The subset of the configurations.
		"""
		return configs.subget(*self.defining_parameters)
	
	@property
	def filestream(self) -> FileStream:
		"""
		Returns the filestream to be used for caching the data.
		The filestream for a specific type of data is created only once, then the same instance is returned.
		The instance is created by calling the constructor of the filestream class, passing the configurations and the arguments;
		therefore, the filestream class must offer a constructor with the following signature:
		__init__(*args)

		:return: The filestream.
		"""
		return self._filestream