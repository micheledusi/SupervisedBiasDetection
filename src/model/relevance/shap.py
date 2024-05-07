# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	 2024				#
# - - - - - - - - - - - - - - - #

# This module contains the class for computing relevance scores with SHAP.

from datasets import Dataset
import logging

import torch

from model.relevance.base import BaseRelevanceCalculator
from utils.config import Configurations
from utils.const import COL_CLASS, COL_EMBS

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RelevanceFromSHAP(BaseRelevanceCalculator):

	def __init__(self, configs: Configurations):
		"""
		Initializer for the relevance from SHAP class.
		"""
		BaseRelevanceCalculator.__init__(self, configs)

	
	def _extraction(self, dataset: Dataset, input_column: str=COL_EMBS, label_column: str=COL_CLASS) -> torch.Tensor:
		"""
		Extracts the relevance scores from the embeddings, using SHAP.
		"""
		raise NotImplementedError("SHAP relevance extraction is not implemented yet.")
		# TODO