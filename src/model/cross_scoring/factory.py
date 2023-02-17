# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #


from enum import Enum

from model.cross_scoring.base import CrossScorer
from model.cross_scoring.mlm import MLMCrossScorer
from model.cross_scoring.pppl import PPPLCrossScorer


class CrossScoreStrategy(Enum):
	MLM = 'mlm'
	PPPL = 'pppl'


class CrossScorerFactory:

	def __init__(self) -> None:
		raise NotImplementedError("This class cannot be instantiated.")

	@staticmethod
	def create(type: CrossScoreStrategy | str, **kwargs) -> CrossScorer:
		"""
		This method creates a cross-scorer, given its type:
		- MLM ('mlm'): Masked Language Model.
		- PPPL ('pppl'): Perplexity.
		
		:param type: The type of the cross-scorer.
		:param kwargs: The arguments to be passed to the cross-scorer constructor.
		:return: The cross-scorer.
		"""
		if type == CrossScoreStrategy.MLM or type == CrossScoreStrategy.MLM.value:
			return MLMCrossScorer(**kwargs)
		elif type == CrossScoreStrategy.PPPL or type == CrossScoreStrategy.PPPL.value:
			return PPPLCrossScorer(**kwargs)