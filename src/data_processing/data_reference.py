# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi	 	#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# Simple classes that embed the data references for the properties.

from data_processing.pattern import PP_PATTERN, SP_PATTERN


class PropertyDataReference:
	"""
	A class representing a reference to a property.
	It contains the following information:
		- The name of the property.
		- The type of the property. Must be "protected" or "stereotyped".
		- The id of the words file to use. (e.g. 01, 02, etc.)
		- The id of the templates file to use. (e.g. 01, 02, etc.)
	"""
	def __init__(self, name: str, type: str, words_file_id: int, templates_file_id: int):
		self.name = name
		self.type = type
		self.words_file_id = words_file_id
		self.templates_file_id = templates_file_id
	
	@property
	def pattern(self) -> str:
		"""
		The pattern to use for the property.

		:return: The pattern to use for the property.
		"""
		if self.type == "protected":
			return PP_PATTERN
		elif self.type == "stereotyped":
			return SP_PATTERN
		else:
			raise ValueError(f"Invalid property type: {self.type}")
		
	@property
	def words_file(self) -> str:
		"""
		The path to the words file.
		"""
		return f"data/{self.type}-p/{self.name}/words-{self.words_file_id:02d}.csv"
	
	@property
	def templates_file(self) -> str:
		"""
		The path to the templates file.
		"""
		return f"data/{self.type}-p/{self.name}/templates-{self.templates_file_id:02d}.csv"


class BiasDataReference:
	"""
	A class representing a reference to a bias, as a couple of properties.
	The bias contains also the ID of the generation file to use.
	"""
	def __init__(self, protected_property: PropertyDataReference, stereotyped_property: PropertyDataReference, generation_id: int):
		self.protected_property = protected_property
		self.stereotyped_property = stereotyped_property
		self.generation_id = generation_id
		