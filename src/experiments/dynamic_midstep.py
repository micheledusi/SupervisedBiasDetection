# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2024				#
# - - - - - - - - - - - - - - - #

# This module is responsible for a general experiment.
# In this experiment, we try several strategies to choose the best value of the midstep a priori.

import logging
from datasets import Dataset
import torch

from experiments.base import Experiment
from model.classification.base import AbstractClassifier
from model.classification.chi import ChiSquaredTest
from model.classification.factory import ClassifierFactory
from utils.config import Configurations


class DynamicPipelineExperiment(Experiment):
	"""
	In this experiment, we train a classifier on the embeddings of the protected property.
	Then, we simply measure its accuracy and F1 score on the test-set.
	"""

	def __init__(self, configs: Configurations) -> None:
		super().__init__(
			"dynamic_pipeline", 
			required_kwargs=[
				self.PROTECTED_EMBEDDINGS_DATASETS_LIST_KEY, 
				self.STEREOTYPED_EMBEDDINGS_DATASETS_LIST_KEY,
			], 
			configs=configs
			)
	
	def _execute(self, **kwargs) -> None:	
		# Getting the embeddings
		prot_embs_ds_list: list[Dataset] = self._extract_value_from_kwargs(self.PROTECTED_EMBEDDINGS_DATASETS_LIST_KEY, **kwargs)
		ster_embs_ds_list: list[Dataset] = self._extract_value_from_kwargs(self.STEREOTYPED_EMBEDDINGS_DATASETS_LIST_KEY, **kwargs)

		assert len(prot_embs_ds_list) == len(ster_embs_ds_list), "The number of protected and stereotyped embeddings datasets must be the same, i.e. the same number of testcases."

		# Phase 1: the reduction step
		# TODO: The reduction step


		# Phase 2: crossing the protected embeddings with the stereotyped embeddings to measure the bias
		chi2 = ChiSquaredTest()
		chi2_values_list: list = []

		# For each dataset/testcase of protected embeddings and the stereotyped embeddings
		for prot_dataset, ster_dataset in zip(prot_embs_ds_list, ster_embs_ds_list):

			# Creating and training the reduction classifier
			classifier: AbstractClassifier = ClassifierFactory.create(self.configs, phase=ClassifierFactory.PHASE_CROSS)
			classifier.train(prot_dataset)

			# We evaluate the predictions of the classifier on the stereotyped embeddings
			ster_dataset_with_predictions: Dataset = classifier.prediction_to_value(classifier.evaluate(ster_dataset))

			# Chi-Squared test
			chi2_results = chi2.test(ster_dataset_with_predictions, "value", "predicted_value")
			chi2_values_list.append(chi2_results)

		chi2_values_list: torch.Tensor = torch.tensor(chi2_values_list)
		chi2_averages = chi2_values_list.mean(dim=0)
		chi2_std_devs = chi2_values_list.std(dim=0)

		# Printing the results
		logging.info("Configurations for the experiment:\n%s", self.configs)
		print(f"RESULTS:                AVG ± STD		over {len(prot_embs_ds_list)} testcases")
		print(f"Chi-Squared value:   {chi2_averages[0]:6.3f} ± {chi2_std_devs[0]:5.3f}")
		print(f"p-value:             {chi2_averages[1]:6.3f} ± {chi2_std_devs[1]:5.3f}")
		print("NOTE: \033[31;1maverage and standard deviation of the p-values are not reliable at all.\033[0m")
		# NOTE: Average and standard deviation are computed over the testcases
		# However, since they come from statistical tests, they are not correct:
		# We cannot combine several p-values to obtain a new p-value with the average and standard deviation operations.
		# Instead, we should do something more advanced, like the Fisher's method. (TODO)