# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	2023				#
# - - - - - - - - - - - - - - - #


from enum import Enum
from typing import Any
from datasets import Dataset

from model.cross_scoring.mlm import MLMCrossScorer
from model.cross_scoring.pppl import PPPLCrossScorer


class CrossScore(Enum):
	MLM = 'mlm'
	PPPL = 'pppl'


class PolarizationStrategy(Enum):
	DIFFERENCE = 'difference'
	RATIO = 'ratio'


DEFAULT_CROSS_SCORE = CrossScore.PPPL
DEFAULT_POLARIZATION_STRATEGY = PolarizationStrategy.DIFFERENCE

AVERAGE_BY_PROTECTED_VALUES = True
AVERAGE_BY_STEREOTYPED_VALUES = False


class CrossBias:

	def __init__(self, cross_score: CrossScore | str = DEFAULT_CROSS_SCORE, polarization: PolarizationStrategy | str = DEFAULT_POLARIZATION_STRATEGY, **kwargs) -> None:
		# Initialize the cross scorer
		if cross_score == CrossScore.MLM or cross_score == CrossScore.MLM.value:
			self._cross_scorer = MLMCrossScorer(**kwargs)
		elif cross_score == CrossScore.PPPL or cross_score == CrossScore.PPPL.value:
			self._cross_scorer = PPPLCrossScorer(**kwargs)

		# Initialize the polarization strategy
		if polarization == PolarizationStrategy.DIFFERENCE or polarization == PolarizationStrategy.DIFFERENCE.value:
			self._polarization_strategy = lambda x, y: x - y
		elif polarization == PolarizationStrategy.RATIO or polarization == PolarizationStrategy.RATIO.value:
			self._polarization_strategy = lambda x, y: x / y

	def __call__(self, templates: Dataset, protected_words: Dataset, stereotyped_words: Dataset) -> Any:
		return self.compute(templates, protected_words, stereotyped_words)

	def compute(self, templates: Dataset, protected_words: Dataset, stereotyped_words: Dataset) -> tuple[tuple[str], tuple[str], Dataset]:
		# Compute the cross scores
		selected_pp_words, selected_sp_words, scores = self._cross_scorer.compute_cross_scores(templates, protected_words, stereotyped_words)
		# "scores" is a tensor with a value for each protected word and each stereotyped word, with shape (n_protected_words, n_stereotyped_words)

		# We average the cross scores according to the protected values
		if AVERAGE_BY_PROTECTED_VALUES:
			pp_entries, scores = self._cross_scorer.average_by_values(selected_pp_words, scores)
		else:
			pp_entries = tuple(selected_pp_words['word'])
		# We average the cross scores according to the stereotyped values
		if AVERAGE_BY_STEREOTYPED_VALUES:
			sp_entries, scores = self._cross_scorer.average_by_values(selected_sp_words, scores)
		else:
			sp_entries = tuple(selected_sp_words['word'])

		# Now "values_scores" is a tensor of different shape, depending on the value of the two flags:
		# - If both flags are True, the shape is (n_protected_values, n_stereotyped_values)
		# - If only the first flag (AVERAGE_BY_PROTECTED_VALUES) is True, the shape is (n_protected_values, n_stereotyped_words)
		# - If only the second flag (AVERAGE_BY_STEREOTYPED_VALUES) is True, the shape is (n_protected_words, n_stereotyped_values)
		# - If both flags are False, the shape is (n_protected_words, n_stereotyped_words)

		# We create a new Dataset
		result: Dataset = Dataset.from_dict({'stereotyped_value': sp_entries})

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