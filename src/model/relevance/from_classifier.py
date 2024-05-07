# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	 2024				#
# - - - - - - - - - - - - - - - #

# This module contains the class for computing relevance scores from a classifier.

from datasets import Dataset
import logging
import torch

from model.classification.base import AbstractClassifier
from model.classification.factory import ClassifierFactory
from model.relevance.base import BaseRelevanceCalculator
from utils.config import Configurations
from utils.const import COL_CLASS, COL_EMBS

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class RelevanceFromClassification(BaseRelevanceCalculator):
	"""
	A class for computing relevance scores from a classifier.
	It uses the property "features_relevance" of the classifier to extract the relevance scores.
	This property is implemented within the classifier, depending on the specific classifier used.
	"""

	def __init__(self, configs: Configurations):
		"""
		Initializer for the relevance from classification class.
		"""
		BaseRelevanceCalculator.__init__(self, configs)
		logger.info("Initializing a calculator of relevance from a classifier.")


	def _extraction(self, dataset: Dataset, input_column: str=COL_EMBS, label_column: str=COL_CLASS) -> torch.Tensor:
		"""
		Extracts the relevance scores from the embeddings, using a classifier.
		"""
		factory = ClassifierFactory(self.configs, phase=ClassifierFactory.PHASE_REDUCTION)
		reduction_classifier: AbstractClassifier = factory.create()
		reduction_classifier.train(dataset, input_column, label_column)
		return reduction_classifier.features_relevance