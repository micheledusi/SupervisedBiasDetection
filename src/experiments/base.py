# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# Base class for the experiments.


from abc import abstractmethod
import time
from datasets import Dataset

from data_processing.sentence_maker import PP_PATTERN, SP_PATTERN
from utils.caching.creation import get_cached_embeddings, get_cached_crossing_scores
from utils.const import NUM_PROC


class Experiment:

	def __init__(self, name: str):
		"""
		The initializer for the experiment class.

		:param name: The name of the experiment.
		"""
		self._name = name
	
	@property
	def name(self) -> str:
		"""
		The name of the experiment.

		:return: The name of the experiment.
		"""
		return self._name
	
	def run(self, **kwargs) -> None:
		"""
		Runs the experiment.
		"""
		start_time = time.time()
		self._execute(**kwargs)
		end_time = time.time()
		print(f"Experiment {self.name} completed in {end_time - start_time} seconds.")
	
	@staticmethod
	def _get_default_embeddings(protected_property: str, stereotyped_property: str, rebuild: bool = False) -> tuple[Dataset, Dataset]:
		protected_words_file = f'data/protected-p/{protected_property}/words-01.csv'
		protected_templates_file = f'data/protected-p/{protected_property}/templates-00.csv'
		protected_embedding_dataset = get_cached_embeddings(protected_property, PP_PATTERN, protected_words_file, protected_templates_file, rebuild=rebuild)
		stereotyped_words_file = f'data/stereotyped-p/{stereotyped_property}/words-01.csv'
		stereotyped_templates_file = f'data/stereotyped-p/{stereotyped_property}/templates-01.csv'
		stereotyped_embedding_dataset = get_cached_embeddings(stereotyped_property, SP_PATTERN, stereotyped_words_file, stereotyped_templates_file, rebuild=rebuild)

		# Preparing embeddings dataset
		def squeeze_embedding_fn(sample):
			sample['embedding'] = sample['embedding'].squeeze()
			return sample
		protected_embedding_dataset = protected_embedding_dataset.map(squeeze_embedding_fn, batched=True, num_proc=NUM_PROC)	# [#words, #templates = 1, #tokens = 1, 768] -> [#words, 768]
		stereotyped_embedding_dataset = stereotyped_embedding_dataset.map(squeeze_embedding_fn, batched=True, num_proc=NUM_PROC)
		return protected_embedding_dataset, stereotyped_embedding_dataset
	
	@staticmethod
	def _get_default_mlm_scores(protected_property: str, stereotyped_property: str) -> Dataset:
		generation_id: int = 1
		return get_cached_crossing_scores(protected_property, stereotyped_property, generation_id)

	@abstractmethod
	def _execute(self, **kwargs) -> None:
		"""
		Description and execution of the core experiment.
		"""
		raise NotImplementedError