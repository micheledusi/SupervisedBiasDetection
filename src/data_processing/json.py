# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi	 	#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module is used to convert data into JSON and vice versa.
# In this project, the "JSON format" is an object expressed as simple Python data structures (lists, dictionaries, strings, numbers, and booleans).
# This way, the JSON format can be dumped into a JSON file, and loaded from it, using the standard Python JSON library.

from abc import ABC, abstractmethod
import numbers
from typing import Any

from datasets import Dataset
import torch

from utils.const import DEVICE


JSON = list | dict[str, Any] | str | numbers.Number | bool
"""
Defines the JSON format as a union of simple Python data structures (lists, dictionaries, strings, numbers, and booleans).
"""

class JSONConverter(ABC):
	"""
	This class is used to convert data to JSON format, and vice versa.
	"""
	def __init__(self) -> None:
		"""
		Initializes the JSONConverter object.
		"""
		super().__init__()

	@abstractmethod
	def to_json(self, data: Any) -> JSON:
		"""
		Converts data to JSON format.
		"""
		raise NotImplementedError("The method 'to_json' has not been implemented. Please, implement it in the derived class.")

	@abstractmethod
	def from_json(self, json_data: JSON) -> Any:
		"""
		Converts data from JSON format.
		"""
		raise NotImplementedError("The method 'from_json' has not been implemented. Please, implement it in the derived class.")
	
	@staticmethod
	def _convert_to_json(data: Any) -> JSON:
		"""
		Converts generic data to JSON format, i.e. to simple Python data structures (lists, dictionaries, strings, numbers, and booleans).
		"""
		if isinstance(data, numbers.Number):
			return data
		elif isinstance(data, str):
			return data
		elif isinstance(data, bool):
			return data
		elif isinstance(data, list | tuple):
			return [JSONConverter._convert_to_json(x) for x in data]
		elif isinstance(data, dict):
			if all(isinstance(k, str) for k in data.keys()):
				return {k: JSONConverter._convert_to_json(v) for k, v in data.items()}
			raise TypeError("The dictionary keys must be strings. Instead they are of type '{}'.".format(type(data.keys()[0])))
		elif isinstance(data, torch.Tensor):
			return data.tolist()
		elif isinstance(data, Dataset):
			return {k: JSONConverter._convert_to_json(v) for k, v in data.to_dict().items()}
		else:
			raise TypeError("The type '{}' is not supported.".format(type(data)))


class EmbeddingsJSONConverter(JSONConverter):
	"""
	This class is used to convert embeddings to JSON format, and vice versa.
	"""
	def to_json(self, embeddings: Dataset) -> JSON:
		# Assert that the embeddings are in the correct format.
		assert "word" in embeddings.features, "The embeddings must have a 'word' column."
		assert "value" in embeddings.features, "The embeddings must have a 'value' column."
		assert "embedding" in embeddings.features, "The embeddings must have an 'embedding' column."
		return JSONConverter._convert_to_json(embeddings)

	def from_json(self, json_data: JSON) -> Dataset:
		return Dataset.from_dict(json_data).with_format("torch", device=DEVICE)


class CrossingScoresJSONConverter(JSONConverter):
	"""
	This class is used to convert crossing scores to JSON format, and vice versa.
	"""
	def to_json(self, crossing_scores: tuple[Dataset, Dataset, torch.Tensor]) -> JSON:
		return JSONConverter._convert_to_json(crossing_scores)


	def from_json(self, json_data: JSON) -> tuple[Dataset, Dataset, torch.Tensor]:
		assert isinstance(json_data, list) and len(json_data) == 3, "The JSON data must be a list of length 3."
		protected_words, stereotyped_words, crossing_scores = tuple(json_data)
		protected_words = Dataset.from_dict(protected_words).with_format("torch", device=DEVICE)
		stereotyped_words = Dataset.from_dict(stereotyped_words).with_format("torch", device=DEVICE)
		crossing_scores = torch.tensor(crossing_scores, device=DEVICE)
		return protected_words, stereotyped_words, crossing_scores


class PolarizationScoresJSONConverter(JSONConverter):
	"""
	This class is used to convert polarization scores to JSON format, and vice versa.
	"""
	def to_json(self, polarization_scores: tuple[tuple[str], tuple[str], Dataset]) -> JSON:
		return JSONConverter._convert_to_json(polarization_scores)

	def from_json(self, json_data: JSON) -> tuple[tuple[str], tuple[str], Dataset]:
		assert isinstance(json_data, list) and len(json_data) == 3, "The JSON data must be a list of length 3."
		protected_entries, stereotyped_entries, polarization_scores = tuple(json_data)
		protected_entries = tuple(protected_entries)
		stereotyped_entries = tuple(stereotyped_entries)
		polarization_scores = Dataset.from_dict(polarization_scores).with_format("torch", device=DEVICE)
		return protected_entries, stereotyped_entries, polarization_scores