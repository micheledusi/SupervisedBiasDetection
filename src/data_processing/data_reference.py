# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi	 	#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# Simple classes that embed the data references for the properties.

from datasets import Dataset
from data_processing.pattern import PATTERN


class PropertyDataReference:
	"""
	A class representing a reference to a property.
	It contains the following information:
		- The name of the property (e.g. "religion", "verb", etc.)
		- The id of the words file to use; it could be a number or a string. (e.g. 01, 02, "three", etc.)
		- The id of the templates file to use. (e.g. 01, 02, etc.)
	"""
	def __init__(self, name: str, words_file_id: int | str, templates_file_id: int | str):
		self.name = name
		self.words_file_id: int | str = words_file_id
		self.templates_file_id: int | str = templates_file_id
	
	@property
	def pattern(self) -> str:
		"""
		The pattern to use for the property.

		@Note: This is an old heritage from the previous version of the code.
		Today, every property has the same pattern, so this property is useless.

		:return: The pattern to use for the property.
		"""
		return PATTERN
	
	def name_with_classes_number(self, values: list | Dataset) -> str:
		"""
		Returns the name of the property with the number of classes attached to it.
		The number of classes is computed from the given list of values, counting the unique ones.

		:param values: The list of values to use to compute the number of classes, or the Dataset in which the 
		'value' column contains the values to use to compute the number of classes.
		:return: The name of the property with the number of classes attached to it.
		"""
		num: int = len(set(values)) if isinstance(values, list) else len(set(values['value']))
		return f"{self.name}{str(num)}"
		
	@property
	def words_file(self) -> str:
		"""
		The path to the words file.
		"""
		suffix: str = self.words_file_id if isinstance(self.words_file_id, str) else f"{self.words_file_id:02d}"
		return f"data/properties/{self.name}/words-{suffix}.csv"
	
	@property
	def templates_file(self) -> str:
		"""
		The path to the templates file.
		"""
		if self.templates_file_id is None or self.templates_file_id == 0 or self.templates_file_id == "":
			return f"data/properties/templates-empty.csv"
		suffix: str = self.templates_file_id if isinstance(self.templates_file_id, str) else f"{self.templates_file_id:02d}"
		return f"data/properties/{self.name}/templates-{suffix}.csv"
	
	def __str__(self) -> str:
		return f"{self.name} ({self.words_file_id}, {self.templates_file_id})"
	
	def __repr__(self) -> str:
		return f"{self.name} ({self.words_file_id}, {self.templates_file_id})"


class BiasDataReference:
	"""
	A class representing a reference to a bias, as a couple of properties.
	The bias contains also the ID of the generation file to use.
	"""
	def __init__(self, protected_property: PropertyDataReference, stereotyped_property: PropertyDataReference, generation_id: int):
		self.protected_property = protected_property
		self.stereotyped_property = stereotyped_property
		self.generation_id = generation_id
		