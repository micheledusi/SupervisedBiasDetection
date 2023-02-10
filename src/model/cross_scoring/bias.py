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
		raw_words_scores = self._cross_scorer.compute_cross_scores(templates, protected_words, stereotyped_words)
		# "raw_words_scores" is a tuple of:
		#	- a dataset of selected protected words
		#	- a dataset of selected stereotyped words
		#	- a tensor of cross scores for each protected word and each stereotyped word, with shape (n_protected_words, n_stereotyped_words)
		# We average the cross scores to properties values
		pp_values, sp_values, values_scores = self._cross_scorer.average_by_values(*raw_words_scores)
		# Now "values_scores" is a tensor of shape (n_protected_values, n_stereotyped_values)

		# We create a new Dataset
		result: Dataset = Dataset.from_dict({'stereotyped_value': sp_values})

		# Compute the bias for each couple of protected values
		for i, pv_i in enumerate(pp_values):
			for j, pv_j in enumerate(pp_values):
				# If the two values are the same, we skip the computation
				if i == j:
					continue
				# Compute the bias
				polarization = self._polarization_strategy(values_scores[i], values_scores[j]).tolist()
				# Add the bias to the result
				result = result.add_column(f'polarization_{pv_i}_{pv_j}', polarization)

		# Return the computed results
		return pp_values, sp_values, result.with_format('pytorch')