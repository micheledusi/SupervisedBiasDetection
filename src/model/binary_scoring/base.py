# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	2023				#
# - - - - - - - - - - - - - - - #
	

from datasets import Dataset
import torch
from model.binary_scoring.crossing.base import CrossingScorer
from model.binary_scoring.crossing.factory import CrossingFactory
from model.binary_scoring.polarization.base import PolarizationScorer
from model.binary_scoring.polarization.factory import PolarizationFactory
from utils.config import Configurations
from utils.const import DEVICE


AVERAGE_BY_PROTECTED_VALUES = True
AVERAGE_BY_STEREOTYPED_VALUES = False


class BinaryScorer:
	"""
	The "CrossBias" class aggregates two sub-phases:
	- The computing of the cross scores, using a "CrossScorer" object
	- The computation of the polarization, using a "PolarizationStrategy" algorithm
	"""

	def __init__(self, configs: Configurations) -> None:
		# Initialize the cross scorer
		self._crossing: CrossingScorer = CrossingFactory.create(configs)
		self._polarization: PolarizationScorer = PolarizationFactory.create(configs)

	def __call__(self, templates: Dataset, protected_words: Dataset, stereotyped_words: Dataset) -> tuple[tuple[str], tuple[str], Dataset]:
		return self.compute(templates, protected_words, stereotyped_words)

	def compute(self, templates: Dataset, protected_words: Dataset, stereotyped_words: Dataset) -> tuple[tuple[str], tuple[str], Dataset]:
		crossing_outcomes = self._crossing.compute(templates, protected_words, stereotyped_words)
		middle_outcomes = BinaryScorer.prepare_crossing_scores_for_polarization(*crossing_outcomes)
		polarization_outcomes = self._polarization.compute(*middle_outcomes)
		return polarization_outcomes
	
	@staticmethod
	def prepare_crossing_scores_for_polarization(protected_words: Dataset, stereotyped_words: Dataset, scores: torch.Tensor) -> tuple[tuple[str], tuple[str], Dataset]:
		"""
		This method prepares the crossing scores for the polarization computation, by averaging the scores according to the protected and stereotyped values.

		:param protected_words: The protected words dataset.
		:param stereotyped_words: The stereotyped words dataset.
		:param scores: The crossing scores tensor, with shape (n_protected_words, n_stereotyped_words).
		:return: A tuple containing:
		- The sorted protected values (as a tuple).
		- The sorted stereotyped values (as a tuple).
		- The average scores tensor, with size (#protected_values, #stereotyped_values).
		"""
		# We average the cross scores according to the protected values
		if AVERAGE_BY_PROTECTED_VALUES:
			pp_entries, scores = BinaryScorer.average_by_values(protected_words, scores)
		else:
			pp_entries = tuple(protected_words['word'])
		# We average the cross scores according to the stereotyped values
		if AVERAGE_BY_STEREOTYPED_VALUES:
			sp_entries, scores = BinaryScorer.average_by_values(stereotyped_words, scores)
		else:
			sp_entries = tuple(stereotyped_words['word'])

		return pp_entries, sp_entries, scores
	
	@staticmethod
	def average_by_values(words: Dataset, words_scores: torch.Tensor, dim: int = 0) -> tuple[tuple[str], torch.Tensor]:
		"""
		This method computes the average of the cross-scores for each pair of words, 
		grouped by the values of the words along a given dimension.

		It's useful to compute the average cross-scores for each protected value, instead of considering all the protected words indipendently.
		It can also be used for the stereotyped values.

		:param words: The words dataset, associated with values to be used for grouping.
		:param words_scores: The cross-scores tensor, to be grouped by values along a given dimension.
		:param dim: The dimension along which the values are grouped.
		:return: A tuple containing:
		- The sorted values (as a tuple).
		- The average scores tensor, with size (#protected_values, #stereotyped_values).
		"""
		# Extracting the values from the dataset
		values = tuple(set(words['value']))

		# Preparing the tensor with the same size of the cross-scores tensor, except for the dimension along which the values are grouped
		size = list(words_scores.shape)
		assert dim < len(size), "The dimension along which the values are grouped must be less than the number of dimensions of the cross-scores tensor."
		assert size[dim] == len(words), "The number of words must be equal to the size of the dimension along which the values are grouped."
		# Setting the size of the dimension along which the values are grouped to the number of values
		size[dim] = len(values)
		average_scores = torch.zeros(size=size).to(DEVICE)

		# Moving the dimension along which the values are grouped to the first dimension, for the resulting tensor
		average_scores = average_scores.swapaxes(dim, 0)

		for val_index, val in enumerate(values):
			# Computing the average of the cross-scores for each pair of words, grouped by the values of the words along a given dimension
			# For each protected value, we compute the average of the cross-scores for each stereotyped value
			pw_indices = [i for i, x in enumerate(words['value']) if x == val]
			selected_scores = words_scores.index_select(dim=0, index=torch.tensor(pw_indices).to(DEVICE))
			average_scores[val_index] = torch.mean(selected_scores, dim=0)
	
		average_scores = average_scores.swapaxes(0, dim)
		return values, average_scores