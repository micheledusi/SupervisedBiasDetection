# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	2023				#
# - - - - - - - - - - - - - - - #


from datasets import Dataset
import torch


class PolarizationScorer:

	STEREOTYPED_ENTRIES_COLUMN: str = 'stereotyped_entry'

	def __init__(self, binary_operator) -> None:
		"""
		This class computes the polarization of a dataset, given a binary operator.
		The binary operator must be a function that takes two tensors as input and returns a tensor as output.
		The input tensors must have the same shape, and the output tensor must have the same shape as the input tensors.
		
		:param binary_operator: The binary operator.
		"""
		self._polarization_strategy = binary_operator

	def __call__(self, protected_words: Dataset, stereotyped_words: Dataset, scores: torch.Tensor) -> tuple[tuple[str], tuple[str], Dataset]:
		return self.compute(protected_words, stereotyped_words, scores)

	def compute(self, protected_entries: tuple[str], stereotyped_entries: tuple[str], scores: torch.Tensor) -> tuple[tuple[str], tuple[str], Dataset]:
		"""
		This method computes the polarization of a dataset, given a binary operator.
		The binary operator must be a function that takes two tensors as input and returns a tensor as output.
		It is applied for each pair of (different) protected entries. The scores are computed along each stereotyped entry.

		:param protected_entries: The protected entries.
		:param stereotyped_entries: The stereotyped entries.
		:param scores: The scores matrix.
		:return: The protected entries, the stereotyped entries, and the polarization dataset.
		"""
		# "scores" is a tensor of different shape, depending on the value of the two flags:
		# - If both flags are True, the shape is (n_protected_values, n_stereotyped_values)
		# - If only the first flag (AVERAGE_BY_PROTECTED_VALUES) is True, the shape is (n_protected_values, n_stereotyped_words)
		# - If only the second flag (AVERAGE_BY_STEREOTYPED_VALUES) is True, the shape is (n_protected_words, n_stereotyped_values)
		# - If both flags are False, the shape is (n_protected_words, n_stereotyped_words)

		# We create a new Dataset
		result: Dataset = Dataset.from_dict({PolarizationScorer.STEREOTYPED_ENTRIES_COLUMN: stereotyped_entries})

		# Compute the bias for each couple of protected entries
		for i, pv_i in enumerate(protected_entries):
			for j, pv_j in enumerate(protected_entries):
				# If the two values are the same, we skip the computation
				if i == j:
					continue
				# Compute the bias
				polarization = self._polarization_strategy(scores[i], scores[j]).tolist()
				# Add the bias to the result
				result = result.add_column(f'polarization_{pv_i}_{pv_j}', polarization)

		# Return the computed results
		return protected_entries, stereotyped_entries, result.with_format('pytorch')