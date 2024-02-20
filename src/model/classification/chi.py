# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi	 	#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module offers a simple implementation of the Chi-Squared test.

from datasets import Dataset
from scipy import stats
import torch


class ChiSquaredTest:
    
	def __init__(self, verbose: bool = False) -> None:
		self._verbose = verbose

	def _count_observed(self, samples_x1: list, samples_x2: list) -> tuple[tuple, tuple, torch.Tensor]:
		"""
		Counts the number of occurrences of each value in the two columns.

		:param values1: The values of the first column.
		:param values2: The values of the second column.
		:return: A dictionary with the number of occurrences of each value in the two columns.
		"""
		assert len(samples_x1) == len(samples_x2), "The two columns must have the same length."
		classes_x1 = tuple(set(samples_x1))
		classes_x2 = tuple(set(samples_x2))
		occurrences = torch.zeros(size=(len(classes_x1), len(classes_x2)), dtype=torch.int)
		for val1, val2 in zip(samples_x1, samples_x2):
			occurrences[classes_x1.index(val1), classes_x2.index(val2)] += 1
		return classes_x1, classes_x2, occurrences
	
	def _print_table(self, title: str, classes_x1: tuple, classes_x2: tuple, table: torch.Tensor) -> None:
		"""
		Prints the observed frequencies.

		:param title: The title of the table.
		:param classes_x1: The classes of the first column.
		:param classes_x2: The classes of the second column.
		:param table: A generic table. It can be the observed frequencies or the expected frequencies.
		"""
		print(title)
		print(f"{'':20s}\t", end="")
		for c2 in classes_x2:
			print(f"{str(c2):20s}\t", end="")
		print()
		for c1, row in zip(classes_x1, table):
			print(f"{str(c1):20s}\t", end="")
			for c2 in row:
				print(f"{str(c2.item()):20s}\t", end="")
			print()
		print()

	def _compute_expected(self, occurrences: torch.Tensor) -> torch.Tensor:
		"""
		Computes the expected frequencies.

		:param occurrences: The observed frequencies.
		:return: The expected frequencies.
		"""
		expected = torch.zeros(size=occurrences.size(), dtype=torch.float)
		row_sum: torch.Tensor = torch.sum(occurrences, dim=1)
		col_sum: torch.Tensor = torch.sum(occurrences, dim=0)
		# Compute the total number of occurrences
		total = torch.sum(occurrences)
		# Compute the expected frequencies
		for i in range(occurrences.size(0)):
			for j in range(occurrences.size(1)):
				expected[i, j] = (row_sum[i] * col_sum[j]) / total
		return expected

	def test(self, dataset: Dataset, x1: str, x2: str) -> tuple[float, float]:
		"""
		Performs the Chi-Squared test on the given dataset.

		:param dataset: The dataset to test.
		:param x1: The name of the first column, i.e. the first categorical variable.
		:param x2: The name of the second column, i.e. the second categorical variable.
		:return: A tuple containing the Chi-Squared statistic and the p-value.
		"""
		# Count the number of occurrences of each value in the two columns
		classes_x1, classes_x2, observed = self._count_observed(dataset[x1], dataset[x2])
		# print(classes_x1, classes_x2, observed)
		# Compute the expected frequencies
		expected = self._compute_expected(observed)
		# Compute the Chi-Squared statistic
		chi_squared = torch.sum(torch.pow(observed - expected, 2) / expected)
		# Compute the p-value
		freedom_degrees = (len(classes_x1) - 1) * (len(classes_x2) - 1)
		p_value = stats.distributions.chi2.sf(chi_squared, freedom_degrees)

		if self._verbose:
			self._print_table("Observed frequencies:", classes_x1, classes_x2, observed)
			self._print_table("Expected frequencies:", classes_x1, classes_x2, expected)
			print(f"Chi-Squared statistic: {chi_squared.item()}")
			print(f"p-value: {p_value}")
		
		return chi_squared.item(), p_value
	

class FisherCombinedProbabilityTest:
	"""
	Performs the Fisher's combined probability test.
	"""

	def __init__(self) -> None:
		pass

	def test(self, p_values: torch.Tensor) -> tuple[float, float, float]:
		"""
		Performs the Fisher's combined probability test on the given p-values. 
		This method is typically applied to a collection of independent test statistics, 
		usually from separate studies having the same null hypothesis. 

		The Fisher's method combines the original p-values into one statistic (X^2), 
		which has a chi-squared distribution with 2k degrees of freedom, 
		where k is the number of original p-values.
		The formula to compute the combined p-value is:
		X^2 = -2 * sum(ln(p_i))

		The null hypothesis of the combined test is that all the original null hypotheses are true.
		Rejecting the null hypothesis  of the combined test means that at least one of the original null hypotheses is rejected, 
		meaning that at least one of the original tests is significant.

		:param p_values: The p-values to combine.
		:return: The combined results: X^2, the degrees of freedom, and the p-value of the combined test.
		"""
		# Compute the combined statistic
		combined_statistic = -2 * torch.sum(torch.log(p_values))
		# Compute the degrees of freedom
		degrees_of_freedom = 2 * len(p_values)
		# Compute the combined p-value
		combined_p_value = stats.distributions.chi2.sf(combined_statistic, degrees_of_freedom)
		return combined_statistic.item(), degrees_of_freedom, combined_p_value
	

class HarmonicMeanPValue():
	"""
	Computes the harmonic mean of the p-values.
	This is a statistical technique to combine the p-values of multiple tests.

	The weighted harmonic mean is defined as:
	H = sum_{i=1}^{n} w_i / sum_{i=1}^{n} w_i / p_i
	where w_i is the weight of the i-th p-value.

	When the weights are all equal, the harmonic mean is:
	H = n / sum_{i=1}^{n} 1 / p_i
	"""

	def __init__(self) -> None:
		pass

	def test(self, p_values: torch.Tensor, weights: torch.Tensor=None) -> float:
		"""
		Computes the harmonic mean of the given p-values.

		:param p_values: The p-values to combine.
		:param weights: The weights of the p-values. If not specified, the weights are all equal to 1/n.
		:return: The harmonic mean of the p-values.
		"""
		if weights is None:
			weights = torch.ones_like(p_values) / len(p_values)
		# Compute the harmonic mean
		harmonic_mean = torch.sum(weights) / torch.sum(weights / p_values)
		return harmonic_mean.item()