# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module offers a class to perform dimensionality reduction that contains different methods.
# It's called CompositeReducer because it's a composite of different reducers.

import functools
import torch

from model.reduction.base import BaseDimensionalityReducer

class CompositeReducer(BaseDimensionalityReducer):
	"""
	This reducer aggregates multiple dimensionality sub-reducers.
	The reducers must be given in the correct order, with compatible sizes, that is:
    the output of the first reducer must be the input of the second reducer.
	"""

	def __init__(self, reducers: list[BaseDimensionalityReducer]):
		for i in range(1, len(reducers)):
			assert reducers[i - 1].out_dim == reducers[i].in_dim
		self._reducers: list[BaseDimensionalityReducer] = reducers
		super().__init__(reducers[0].out_dim, reducers[-1].in_dim)

	def _reduction_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
		results: torch.Tensor = embeddings
		for reducer in self._reducers:
			print("\t> ", end='')
			results = reducer.reduce(results)
		return results

	def get_transformation_matrix(self) -> torch.Tensor:
		# Using a FOLDLEFT-like function, we can compute the transformation matrix
		# by multiplying the matrices of the sub-reducers.
		return functools.reduce(torch.matmul, [reducer.get_transformation_matrix() for reducer in self._reducers], torch.eye(self.in_dim))

