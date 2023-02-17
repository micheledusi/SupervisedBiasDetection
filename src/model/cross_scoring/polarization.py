# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	2023				#
# - - - - - - - - - - - - - - - #


from enum import Enum
from typing import Any
from datasets import Dataset
import torch

from model.cross_scoring.base import CrossScorer
from model.cross_scoring.factory import CrossScoreStrategy, CrossScorerFactory


class PolarizationStrategy(Enum):
	DIFFERENCE = 'difference'
	RATIO = 'ratio'


DEFAULT_CROSS_SCORE = CrossScoreStrategy.PPPL
DEFAULT_POLARIZATION_STRATEGY = PolarizationStrategy.DIFFERENCE

AVERAGE_BY_PROTECTED_VALUES = True
AVERAGE_BY_STEREOTYPED_VALUES = False


class PolarizationScorer:

	STEREOTYPED_ENTRIES_COLUMN: str = 'stereotyped_entry'

	def __init__(self, strategy: PolarizationStrategy | str = DEFAULT_POLARIZATION_STRATEGY) -> None:
		# Initialize the polarization strategy
		if strategy == PolarizationStrategy.DIFFERENCE or strategy == PolarizationStrategy.DIFFERENCE.value:
			self._polarization_strategy = lambda x, y: x - y
		elif strategy == PolarizationStrategy.RATIO or strategy == PolarizationStrategy.RATIO.value:
			self._polarization_strategy = lambda x, y: x / y

	def __call__(self, protected_words: Dataset, stereotyped_words: Dataset, scores: torch.Tensor) -> tuple[tuple[str], tuple[str], Dataset]:
		return self.compute(protected_words, stereotyped_words, scores)

	def compute(self, protected_words: Dataset, stereotyped_words: Dataset, scores: torch.Tensor) -> tuple[tuple[str], tuple[str], Dataset]:
		# We average the cross scores according to the protected values
		if AVERAGE_BY_PROTECTED_VALUES:
			pp_entries, scores = CrossScorer.average_by_values(protected_words, scores)
		else:
			pp_entries = tuple(protected_words['word'])
		# We average the cross scores according to the stereotyped values
		if AVERAGE_BY_STEREOTYPED_VALUES:
			sp_entries, scores = CrossScorer.average_by_values(stereotyped_words, scores)
		else:
			sp_entries = tuple(stereotyped_words['word'])

		# Now "values_scores" is a tensor of different shape, depending on the value of the two flags:
		# - If both flags are True, the shape is (n_protected_values, n_stereotyped_values)
		# - If only the first flag (AVERAGE_BY_PROTECTED_VALUES) is True, the shape is (n_protected_values, n_stereotyped_words)
		# - If only the second flag (AVERAGE_BY_STEREOTYPED_VALUES) is True, the shape is (n_protected_words, n_stereotyped_values)
		# - If both flags are False, the shape is (n_protected_words, n_stereotyped_words)

		# We create a new Dataset
		result: Dataset = Dataset.from_dict({PolarizationScorer.STEREOTYPED_ENTRIES_COLUMN: sp_entries})

		# Compute the bias for each couple of protected entries
		for i, pv_i in enumerate(pp_entries):
			for j, pv_j in enumerate(pp_entries):
				# If the two values are the same, we skip the computation
				if i == j:
					continue
				# Compute the bias
				polarization = self._polarization_strategy(scores[i], scores[j]).tolist()
				# Add the bias to the result
				result = result.add_column(f'polarization_{pv_i}_{pv_j}', polarization)

		# Return the computed results
		return pp_entries, sp_entries, result.with_format('pytorch')
	

class CrossBias:
	"""
	The "CrossBias" class aggregates two sub-phases:
	- The computing of the cross scores, using a "CrossScorer" object
	- The computation of the polarization, using a "PolarizationStrategy" algorithm
	"""

	def __init__(self, cross_score: CrossScoreStrategy | str = DEFAULT_CROSS_SCORE, polarization: PolarizationStrategy | str = DEFAULT_POLARIZATION_STRATEGY, **kwargs) -> None:
		# Initialize the cross scorer
		self._cross_scorer: CrossScorer = CrossScorerFactory.create(type=cross_score, **kwargs)
		self._polarization: PolarizationScorer = PolarizationScorer(strategy=polarization)

	def __call__(self, templates: Dataset, protected_words: Dataset, stereotyped_words: Dataset) -> Any:
		return self.compute(templates, protected_words, stereotyped_words)

	def compute(self, templates: Dataset, protected_words: Dataset, stereotyped_words: Dataset) -> tuple[tuple[str], tuple[str], Dataset]:
		# Compute the cross scores
		selected_pp_words, selected_sp_words, scores = self._cross_scorer.compute_cross_scores(templates, protected_words, stereotyped_words)
		# "scores" is a tensor with a value for each protected word and each stereotyped word, with shape (n_protected_words, n_stereotyped_words)
		# Now we aggregate (when required) the words by values, and we compute the polarizations pairing the protected entries
		return self._polarization.compute(selected_pp_words, selected_sp_words, scores)