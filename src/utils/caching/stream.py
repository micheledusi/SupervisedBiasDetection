# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi	 	#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module is used to manage the caching of data in human-readable JSON files.

from abc import ABC, abstractmethod
import json
import os
from typing import Any, Type
from data_processing.json import JSONConverter
import pickle as pkl


class FileStream(ABC):
	"""
	This abstract class is used to manage the caching of data.
	"""
	def __init__(self, file_extension: str) -> None:
		"""
		Initializes the FileStream object.
		"""
		self._file_extension = file_extension

	@abstractmethod
	def write(self, filepath: str, data: Any) -> None:
		"""
		Writes data.
		"""
		raise NotImplementedError("The method 'write' has not been implemented. Please, implement it in the derived class.")

	@abstractmethod
	def read(self, filepath: str) -> Any:
		"""
		Reads data.
		"""
		raise NotImplementedError("The method 'read' has not been implemented. Please, implement it in the derived class.")
	
	def delete(self, filepath: str, extension: str = None) -> None:
		"""
		This method deletes a file, given its path and extension.
		If the extension is not provided, the default one is used. (see self.file_extension)
		In this case, the method will add the extension to the file path, if it is not already present.
		Otherwise, the method will use the provided extension and make sure that the file path ends with it.

		:param filepath: The path of the file to be deleted.
		:param extension: The extension of the file to be deleted (optional).
		"""
		filepath_ext = self.add_extension(filepath, extension)
		if os.path.exists(filepath_ext):
			os.remove(filepath_ext)
		else:
			raise FileNotFoundError(f"The file '{filepath_ext}' does not exist.")
	
	def add_extension(self, filepath: str, extension: str = None) -> str:
		"""
		Adds the file extension to the file path, if it is not already present.
		Otherwise, it returns the file path as it is.
		The extension is either the default one (see self.file_extension) or the provided one, if any.

		:param filepath: The path of the file.
		:param extension: The extension of the file (optional).
		"""
		# If the extension is not provided, the default one is used.
		if not extension:
			extension = self.file_extension
		# Adds the extension to the file path, if it is not already present.
		if not filepath.endswith(extension):
			return filepath + extension
		return filepath
	
	@property
	def file_extension(self) -> str:
		"""
		Returns the file extension.
		"""
		return self._file_extension


class PickleFileStream(FileStream):
	"""
	This class is used to manage the caching of data in binary files.
	"""
	def __init__(self) -> None:
		"""
		Initializes the PickleFileStream object.
		"""
		super().__init__('.pkl')

	def write(self, filepath: str, data: Any) -> None:
		"""
		Writes data in a binary file.
		"""
		filepath_ext = self.add_extension(filepath)
		pkl.dump(data, open(filepath_ext, 'wb'))

	def read(self, filepath: str) -> Any:
		"""
		Reads data from a binary file.
		"""
		filepath_ext = self.add_extension(filepath)
		return pkl.load(open(filepath_ext, 'rb'))


class JSONFileStream(FileStream):
	"""
	This class is used to manage the caching of data in human-readable JSON files.
	"""
	def __init__(self, json_converter_cls: Type[JSONConverter]) -> None:
		"""
		Initializes the JSONFileStream object.
		"""
		super().__init__('.json')
		# Initializes the JSON converter object with the given configurations.
		self.json_converter = json_converter_cls()

	def write(self, filepath: str, data: Any) -> None:
		"""
		Writes data in a JSON file.
		"""
		json_data = self.json_converter.to_json(data)
		filepath_ext = self.add_extension(filepath)
		with open(filepath_ext, 'w') as f:
			json.dump(json_data, f, indent=4)

	def read(self, filepath: str) -> Any:
		"""
		Reads data from a JSON file.
		"""
		filepath_ext = self.add_extension(filepath)
		with open(filepath_ext, 'r') as f:
			json_data = json.load(f)
			return self.json_converter.from_json(json_data)
		
