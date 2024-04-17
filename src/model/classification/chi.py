# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi	 	#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module offers a simple implementation of the Chi-Squared test.

import logging
import re
from colorist import Color
from datasets import Dataset
from scipy import stats
from tabulate import tabulate
import torch

from utils.const import ANSI_FILTER


COLOR_PREDICTED: str = Color.RED
COLOR_ACTUAL: str = Color.MAGENTA
COLOR_TITLE: str = Color.YELLOW
STR_PREDICTED: str = COLOR_PREDICTED + "PREDICTED" + Color.OFF
STR_ACTUAL: str = COLOR_ACTUAL + "ACTUAL" + Color.OFF

ContingencyTable = tuple[tuple, tuple, torch.Tensor]


class ChiSquaredTest:
    
	def __init__(self, verbose: bool = False) -> None:
		self._verbose = verbose
		self._contingency_table = None
	

	@property
	def contingency_table(self) -> ContingencyTable:
		if self._contingency_table is None:
			raise ValueError("The contingency table has not been computed yet.")
		return self._contingency_table


	def _count_observed(self, samples_x1: list, samples_x2: list) -> ContingencyTable:
		"""
		Counts the number of occurrences of each value in the two columns.

		:param values1: The values of the first column.
		:param values2: The values of the second column.
		:return: A dictionary with the number of occurrences of each value in the two columns.
		"""
		assert len(samples_x1) == len(samples_x2), "The two columns must have the same length."
		classes_x1 = tuple(sorted(set(samples_x1)))
		classes_x2 = tuple(sorted(set(samples_x2)))
		occurrences = torch.zeros(size=(len(classes_x1), len(classes_x2)), dtype=torch.int)
		for val1, val2 in zip(samples_x1, samples_x2):
			occurrences[classes_x1.index(val1), classes_x2.index(val2)] += 1
		return classes_x1, classes_x2, occurrences
	

	def get_formatted_table(self, title: str, table: ContingencyTable = None, use_latex: bool = False) -> str:
		"""
		Prints the observed frequencies.

		:param title: The title of the table.
		:param table: The contingency table to print. If not specified, the table is computed from the observed frequencies.
		:param is_latex: Whether to format the table in LaTeX.
		"""
		# Unpack the table
		if not table:
			classes_x1, classes_x2, data_table = self.contingency_table
		else:
			classes_x1, classes_x2, data_table = table

		table_str: list[list] = []
		# Header
		title_str = COLOR_TITLE + title + Color.OFF
		table_str.append([title_str, ""] + [STR_PREDICTED] * len(classes_x2))
		table_str.append(["", ""] + [COLOR_PREDICTED + str(c) + Color.OFF for c in classes_x2])
		# Body
		for c1, row in zip(classes_x1, data_table):
			table_str.append([STR_ACTUAL, COLOR_ACTUAL + str(c1) + Color.OFF] + [f"{elem.item():.2f}" for elem in row])

		# Print the table
		if use_latex:
			latex_text: str = tabulate(table_str, headers="firstrow", tablefmt="latex", numalign="center", stralign="center")
			return ANSI_FILTER.sub("", latex_text)
		else:
			return tabulate(table_str, headers="firstrow", tablefmt="rounded_grid", numalign="center", stralign="center")


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


	def test_from_dataset(self, dataset: Dataset, x1: str, x2: str) -> tuple[float, float]:
		"""
		Performs the Chi-Squared test on the given dataset.

		:param dataset: The dataset to test.
		:param x1: The name of the first column, i.e. the first categorical variable.
		:param x2: The name of the second column, i.e. the second categorical variable.
		:return: A tuple containing the Chi-Squared statistic and the p-value.
		"""
		# Count the number of occurrences of each value in the two columns
		self._contingency_table = self._count_observed(dataset[x1], dataset[x2])
		# The observed frequencies are stored in the contingency table
		# We now complete the test on the contingency table
		return self.__test()	


	def test_from_contingency_table(self, contingency_table: ContingencyTable) -> tuple[float, float]:
		"""
		Performs the Chi-Squared test on the given contingency table.

		:param contingency_table: The contingency table containing the observed frequencies.
		:return: A tuple containing the Chi-Squared statistic and the p-value.
		"""
		self._contingency_table = contingency_table
		# The observed frequencies are stored in the contingency table
		# We now complete the test on the contingency table
		return self.__test()


	def __test(self) -> tuple[float, float]:
		"""
		Performs the Chi-Squared test on the observed frequencies.
		This method assumes that the observed frequencies are already stored in the contingency table.

		:return: A tuple containing the Chi-Squared statistic and the p-value.
		"""
		# Unpack the table
		classes_x1, classes_x2, observed = self.contingency_table

		# Compute the expected frequencies
		expected = self._compute_expected(observed)
		# Compute the Chi-Squared statistic
		chi_squared = torch.sum(torch.pow(observed - expected, 2) / expected)
		# Compute the p-value
		freedom_degrees = (len(classes_x1) - 1) * (len(classes_x2) - 1)
		p_value = stats.distributions.chi2.sf(chi_squared, freedom_degrees)

		if self._verbose:
			print(self.get_formatted_table("OBSERVED:", self.contingency_table))
			# self._print_table("Expected frequencies:", classes_x1, classes_x2, expected)
			print(f"Chi-Squared statistic: {chi_squared.item()}")
			print(f"p-value: {p_value}")
		
		return chi_squared.item(), p_value
	

	@staticmethod
	def average_contingency_matrices(matrices: list[ContingencyTable]) -> ContingencyTable:
		"""
		Averages the given contingency tables.

		:param matrices: The contingency tables to average.
		:return: The average contingency table.
		"""
		# Compute the classes
		classes_1 = []
		classes_2 = []
		for cls_1, cls_2, _ in matrices:
			classes_1.extend(cls_1)
			classes_2.extend(cls_2)
		classes_1 = tuple(sorted(set(classes_1)))
		classes_2 = tuple(sorted(set(classes_2)))
		logging.debug(f"Classes of the first column:  {classes_1}")
		logging.debug(f"Classes of the second column: {classes_2}")

		sum_tensor: torch.Tensor = torch.zeros(size=(len(classes_1), len(classes_2)), dtype=torch.float)

		for cls_1, cls_2, tensor in matrices:
			# We cannot assume that the classes are the same in all tables
			# We need to map the classes to the average tensor
			for i, c1 in enumerate(cls_1):
				for j, c2 in enumerate(cls_2):
					sum_tensor[classes_1.index(c1), classes_2.index(c2)] += tensor[i, j]

		# Compute the average matrix
		average_tensor = sum_tensor / len(matrices)
		return classes_1, classes_2, average_tensor
	

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
	https://en.wikipedia.org/wiki/Harmonic_mean_p-value

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