# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2024				#
# - - - - - - - - - - - - - - - #

# This module is responsible for a general experiment.
# In this experiment, we try several strategies to choose the best value of the midstep a priori.

from enum import Enum
import logging
from colorist import Color, Effect
from datasets import Dataset
import torch
from tqdm import tqdm

from experiments.base import Experiment
from model.classification.base import AbstractClassifier
from model.classification.chi import ChiSquaredTest, FisherCombinedProbabilityTest, HarmonicMeanPValue
from model.classification.factory import ClassifierFactory
from model.embedding.cluster_validator import ClusteringScorer
from model.reduction.weights import WeightsSelectorReducer
from utils.config import Configurations

EMB_COL: str = "embedding"


class ReductionStrategy(Enum):
	NONE = "none"
	RANDOM_SAMPLING = "random_sampling"
	RELEVANT_SAMPLING = "relevant_sampling"
	ANTI_RELEVANT_SAMPLING = "anti_relevant_sampling"
	TOP_N_WITH_BEST_CHOICE = "top_n_with_best_choice"
	ANTI_TOP_N_WITH_BEST_CHOICE = "anti_top_n_with_best_choice"


class DynamicPipelineExperiment(Experiment):
	"""
	In this experiment, we try several strategies to choose the best value of the midstep a priori.
	"""

	DatasetPair = tuple[Dataset, Dataset]

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
		reduced_embs_ds_by_strategy: dict[ReductionStrategy, list[tuple[Dataset, Dataset]]] = {strategy: [] for strategy in ReductionStrategy}

		for prot_embs_ds, ster_embs_ds in tqdm(zip(prot_embs_ds_list, ster_embs_ds_list), desc="Reduction step", total=len(prot_embs_ds_list)):
			reduction_results = self._execute_reduction(prot_embs_ds, ster_embs_ds)
			# We append the results to the list of datasets for each strategy
			for strategy, reduced_embs_ds in reduction_results.items():
				reduced_embs_ds_by_strategy[strategy].append(reduced_embs_ds)
		
		# Phase 2: crossing the protected embeddings with the stereotyped embeddings to measure the bias
		for strategy, reduced_embs_ds_list in reduced_embs_ds_by_strategy.items():
			# print(f"{Color.GREEN}Strategy: {strategy}{Color.OFF}")
			strategy_results: dict = self._execute_crossing(reduced_embs_ds_list)
			strategy_results['strategy'] = strategy.value
			self.results_collector.collect(self.configs, strategy_results)


	def _execute_reduction(self, prot_embs_ds: Dataset, ster_embs_ds: Dataset) -> dict[ReductionStrategy, DatasetPair]:
		"""
		Executes the reduction step, with different strategies.

		:param prot_embs_ds: The dataset of the protected embeddings.
		:param ster_embs_ds: The dataset of the stereotyped embeddings.
		:return: A dictionary containing the datasets of the protected and stereotyped embeddings after the reduction step.
		Each key represents the strategy used for the reduction.
		"""
		reduced_embs: dict[str, tuple[Dataset, Dataset]] = {}

		# No reduction
		reduced_embs[ReductionStrategy.NONE] = (prot_embs_ds, ster_embs_ds)

		# Random sampling
		# Select a random subset of the features
		# We need an array of equal probabilities, one for each feature
		equal_probabilities = 0.5 * torch.ones(size=(prot_embs_ds[EMB_COL].shape[-1],))
		reduced_embs[ReductionStrategy.RANDOM_SAMPLING] = self._sample_with_probability(prot_embs_ds, ster_embs_ds, equal_probabilities)

		# Getting the relevance of the features
		relevance: torch.Tensor = self._compute_relevance(prot_embs_ds)
		relevance_probabilities = relevance / relevance.max()

		# For each feature, we sample it with a probability equal to its relevance
		reduced_embs[ReductionStrategy.RELEVANT_SAMPLING] = self._sample_with_probability(prot_embs_ds, ster_embs_ds, relevance_probabilities)

		# For each feature, we sample it with a probability equal to 1 - its relevance
		reduced_embs[ReductionStrategy.ANTI_RELEVANT_SAMPLING] = self._sample_with_probability(prot_embs_ds, ster_embs_ds, 1 - relevance_probabilities)

		# We choose the best value for N
		best_n: int = self._choose_best_n(prot_embs_ds, relevance)
		# We select the N features with the highest relevance
		top_n_relevant_features_reducer = WeightsSelectorReducer.from_weights(relevance, best_n)
		reduced_embs[ReductionStrategy.TOP_N_WITH_BEST_CHOICE] = top_n_relevant_features_reducer.reduce_ds(prot_embs_ds), top_n_relevant_features_reducer.reduce_ds(ster_embs_ds)

		# We select the N features with the lowest relevance
		anti_relevance = 1 / relevance
		anti_top_n_relevant_features_reducer = WeightsSelectorReducer.from_weights(anti_relevance, best_n)
		reduced_embs[ReductionStrategy.ANTI_TOP_N_WITH_BEST_CHOICE] = anti_top_n_relevant_features_reducer.reduce_ds(prot_embs_ds), anti_top_n_relevant_features_reducer.reduce_ds(ster_embs_ds)

		return reduced_embs


	def _choose_best_n(self, prot_embs_ds: Dataset, relevance: torch.Tensor) -> int:
		"""
		Chooses the best value for N, i.e. the number of features to keep after the reduction step.

		:param prot_embs_ds: The dataset of the protected embeddings.
		:return: The best value for N.
		"""
		logging.debug(f"Relevance shape: {relevance.shape}")
		possible_n_values = range(1, prot_embs_ds[EMB_COL].shape[-1] + 1)
		logging.debug(f"Possible N values: {possible_n_values}")
		scorer: ClusteringScorer = ClusteringScorer(self.configs)
		results = {n: self._compute_clustering_score(prot_embs_ds, n, relevance, scorer) for n in possible_n_values}
		
		# # DEBUG
		# x = list(results.keys())
		# y = list(results.values())
		# import matplotlib.pyplot as plt
		# fig, ax = plt.subplots()
		# ax.plot(x, y)
		# ax.set(xlabel='N', ylabel='Separation score', title='Separation score for different values of N')
		# ax.grid()
		# plt.show()

		best_n = max(results, key=results.get)
		return best_n

	
	def _compute_clustering_score(self, prot_embs_ds: Dataset, n: int, relevance: torch.Tensor, scorer: ClusteringScorer) -> float:
		"""
		Computes the separation score for the given value of N.
		The higher the score, the more the embeddings are separated in classes.

		:param prot_embs_ds: The dataset of the protected embeddings.
		:param n: The value of N to use for the reduction.
		:param relevance: The relevance of the features.
		:param scorer: The scorer to use for the computation of distances.
		:return: The score for the given value of N.
		"""
		reducer = WeightsSelectorReducer.from_weights(relevance, n)
		reduced_prot_embs_ds = reducer.reduce_ds(prot_embs_ds)
		return scorer.compute_clustering_score(reduced_prot_embs_ds, "value", EMB_COL)


	def _compute_relevance(self, prot_embs_ds: Dataset) -> torch.Tensor:
		"""
		Computes the relevance of the features, for the embeddings in the dataset.

		:param prot_embs_ds: The dataset of the protected embeddings.
		:return: A tensor containing the relevance of each feature for the embeddings in the dataset.
		"""
		reduction_classifier: AbstractClassifier = ClassifierFactory.create(self.configs, phase=ClassifierFactory.PHASE_REDUCTION)
		reduction_classifier.train(prot_embs_ds)
		return reduction_classifier.features_relevance
	

	def _sample_with_probability(self, prot_embs_ds: Dataset, ster_embs_ds: Dataset, probabilities: torch.Tensor) -> tuple[Dataset, Dataset]:
		"""
		Samples the embeddings in the dataset with a probability equal to the given array.

		:param prot_embs_ds: The dataset of the protected embeddings.
		:param ster_embs_ds: The dataset of the stereotyped embeddings.
		:param probabilities: The probabilities to use for the sampling.
		:return: The datasets of embeddings after the sampling.
		"""
		mask = torch.rand(size=(len(probabilities),)) < probabilities
		logging.debug(f"Mask shape: {mask.shape}")
		indices = torch.nonzero(mask).squeeze()
		logging.debug(f"Indices shape: {indices.shape}")
		prot_embs: torch.Tensor = prot_embs_ds[EMB_COL]
		ster_embs: torch.Tensor = ster_embs_ds[EMB_COL]
		logging.debug(f"Prot embs shape: {prot_embs.shape}")
		logging.debug(f"Ster embs shape: {ster_embs.shape}")
		reduced_prot_embs_ds: Dataset = prot_embs_ds\
			.remove_columns(EMB_COL)\
			.add_column(EMB_COL, prot_embs[:, indices].tolist())
		reduced_ster_embs_ds: Dataset = ster_embs_ds\
			.remove_columns(EMB_COL)\
			.add_column(EMB_COL, ster_embs[:, indices].tolist())
		logging.debug(f"Reduced prot embs shape: {(reduced_prot_embs_ds[EMB_COL]).shape}")
		logging.debug(f"Reduced ster embs shape: {(reduced_ster_embs_ds[EMB_COL]).shape}")
		return reduced_prot_embs_ds, reduced_ster_embs_ds


	def _execute_crossing(self, embs_ds_list: list[tuple[Dataset, Dataset]]) -> dict[str, float]:
		"""
		Compares the protected embeddings with the stereotyped embeddings to measure the bias.
		The comparison is done using the chi-squared test for each testcase.
		Then, the results are combined using the Fisher's method and the Harmonic mean.

		:param embs_ds_list: The list of datasets of the embeddings: each element is a tuple containing the protected and the stereotyped embeddings datasets.
		:return: A dictionary containing the results of the comparison.
		"""
		chi2 = ChiSquaredTest()
		chi2_values_list: list = []

		# For each dataset/testcase of protected embeddings and the stereotyped embeddings
		for prot_dataset, ster_dataset in embs_ds_list:

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

		# Combining the results with the Fisher's method
		fisher_test = FisherCombinedProbabilityTest()
		x2, degs, p_value = fisher_test.test(chi2_values_list.select(dim=1, index=1))	# We only need the p-values

		# Combining the results with the Harmonic mean
		harmonic_mean = HarmonicMeanPValue()
		harmonic_mean_p_value = harmonic_mean.test(chi2_values_list.select(dim=1, index=1))	# We only need the p-values

		# Printing the results
		logging.info("Configurations for the experiment:\n%s", self.configs)
		# print(f"{Effect.BOLD}AGGREGATED RESULTS{Effect.OFF}:     AVG ± STD		over {len(embs_ds_list)} testcases")
		# print(f"Chi-Squared value:   {chi2_averages[0]:6.3f} ± {chi2_std_devs[0]:5.3f}")
		# print(f"{Effect.BOLD}COMBINED RESULTS{Effect.OFF}:")
		# print(f"Fisher Test value:   {x2:.3f} with {degs} degrees of freedom")
		# print(f"p-value:             {p_value}")
		# print(f"Harmonic mean p-value: {harmonic_mean_p_value}")

		return dict({
			"chi2_avg": chi2_averages[0],
			"chi2_std_dev": chi2_std_devs[0],
			"fisher_x2": x2,
			"fisher_degrees": degs,
			"fisher_p_value": p_value,
			"harmonic_mean_p_value": harmonic_mean_p_value
		})