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
from colorist import Color
from datasets import Dataset
import torch
from tqdm import tqdm

from experiments.base import Experiment
from model.classification.base import AbstractClassifier
from model.classification.chi import ChiSquaredTest, FisherCombinedProbabilityTest, HarmonicMeanPValue
from model.classification.factory import ClassifierFactory
from model.embedding.center import EmbeddingCenterer
from model.embedding.cluster_validator import ClusteringScorer
from model.embedding.combinator import EmbeddingsCombinator
from model.reduction.base import BaseDimensionalityReducer
from model.reduction.pca import TrainedPCAReducer
from model.reduction.weights import WeightsSelectorReducer
from utils.caching.creation import get_cached_raw_embeddings
from utils.config import Configurations
from view.plotter.scatter import DatasetPairScatterPlotter

EMB_COL: str = "embedding"

REBUILD_DATASETS = False

class ReductionStrategy(Enum):
	NONE = "none"
	RANDOM_SAMPLING = "random_sampling"
	RELEVANT_SAMPLING = "relevant_sampling"
	ANTI_RELEVANT_SAMPLING = "anti_relevant_sampling"
	# ENHANCED_RELEVANT_SAMPLING = "enhanced_relevant_sampling"
	# ENHANCED_ANTI_RELEVANT_SAMPLING = "enhanced_anti_relevant_sampling"
	# TOP_N_WITH_BEST_CHOICE = "top_n_with_best_choice" 			@deprecated
	# ANTI_TOP_N_WITH_BEST_CHOICE = "anti_top_n_with_best_choice"	@deprecated


class DynamicPipelineExperiment(Experiment):
	"""
	In this experiment, we try several strategies to choose the best value of the midstep a priori.
	"""

	DatasetPair = tuple[Dataset, Dataset]

	def __init__(self, configs: Configurations) -> None:
		super().__init__(
			"dynamic_pipeline", 
			required_kwargs=['prot_prop', 'ster_prop'], 
			configs=configs
			)


	def _execute(self, **kwargs) -> None:	
		# For every combination of parameters (for the step of the raw embeddings computation)
		for configs_re in self.configs.iterate_over(Configurations.ParametersSelection.RAW_EMBEDDINGS_COMPUTATION):
			
			# Loading the datasets
			protected_property_ds: Dataset = get_cached_raw_embeddings(self.protected_property, configs_re, REBUILD_DATASETS)
			stereotyped_property_ds: Dataset = get_cached_raw_embeddings(self.stereotyped_property, configs_re, REBUILD_DATASETS)

			logging.debug(f"Resulting protected raw dataset for property \"{self.protected_property.name}\":\n{protected_property_ds}")
			logging.debug(f"Resulting stereotyped raw dataset for property \"{self.stereotyped_property.name}\":\n{stereotyped_property_ds}")
		
			# Combining the embeddings
			combinator = EmbeddingsCombinator(configs_re)

			combined_protected_embeddings: dict[Configurations, list[Dataset]] = combinator.combine(protected_property_ds)
			combined_stereotyped_embeddings: dict[Configurations, list[Dataset]] = combinator.combine(stereotyped_property_ds)

			# Centering the embeddings
			# For each configuration of the combined embeddings:
			for key_configs in combined_protected_embeddings:
				centerer: EmbeddingCenterer = EmbeddingCenterer(key_configs)
				# Centering the embeddings for each testcase
				combined_protected_embeddings[key_configs] = [centerer.center(ds) for ds in combined_protected_embeddings[key_configs]]
			for key_configs in combined_stereotyped_embeddings:
				centerer: EmbeddingCenterer = EmbeddingCenterer(key_configs)
				# Centering the embeddings for each testcase
				combined_stereotyped_embeddings[key_configs] = [centerer.center(ds) for ds in combined_stereotyped_embeddings[key_configs]]	

			# Now we have the combined embeddings, we can proceed with the bias detection
			for key_configs in combined_protected_embeddings:
				print(f"Running experiment for configuration:\n{key_configs}")

				# We assume that the keys of the two dictionaries are the same
				# Meaning that the configurations are the same for both the protected and the stereotyped property
				protected_embs_ds_list: list[Dataset] = combined_protected_embeddings[key_configs]
				stereotyped_embs_ds_list: list[Dataset] = combined_stereotyped_embeddings[key_configs]

				self._execute_for_single_configuration(key_configs, protected_embs_ds_list, stereotyped_embs_ds_list)


	def _execute_for_single_configuration(self, current_configs: Configurations, prot_embs_ds_list: list[Dataset], ster_embs_ds_list: list[Dataset]) -> None:
		"""
		Executes the experiment for a single configuration of the embeddings.

		:param prot_embs_ds_list: The list of datasets of the protected embeddings.
		:param ster_embs_ds_list: The list of datasets of the stereotyped embeddings.
		"""
		# Phase 1: the reduction step
		reduced_embs_ds_by_strategy: dict[ReductionStrategy, list[tuple[Dataset, Dataset]]] = {strategy: [] for strategy in ReductionStrategy}

		# This cycle is executed for each testcase
		for prot_embs_ds, ster_embs_ds in tqdm(zip(prot_embs_ds_list, ster_embs_ds_list), desc="Reduction step", total=len(prot_embs_ds_list)):
			# Each testcase is reduced with different strategies
			reduction_results: dict = self._execute_reduction(prot_embs_ds, ster_embs_ds)
			# The result of each strategy is collected, along with the results of the other testcases for the same strategy
			for strategy, reduced_embs_ds in reduction_results.items():
				reduced_embs_ds_by_strategy[strategy].append(reduced_embs_ds)

		# Phase 1.5: plotting the reduced embeddings with the ScatterPlotter
		# For each strategy
		for strategy, reduced_embs_ds_list in reduced_embs_ds_by_strategy.items():
			logging.info(f"Plotting embeddings for STRATEGY: {Color.GREEN}{strategy.name}{Color.OFF}")
			# For each testcase in this strategy
			for prot_embs_ds, ster_embs_ds in reduced_embs_ds_list:
				self._execute_plotting(strategy, prot_embs_ds, ster_embs_ds)

		# FIXME DEBUG
		exit()
		
		# Phase 2: crossing the protected embeddings with the stereotyped embeddings to measure the bias
		for strategy, reduced_embs_ds_list in reduced_embs_ds_by_strategy.items():
			# print(f"{Color.GREEN}Strategy: {strategy}{Color.OFF}")
			strategy_results: dict = self._execute_crossing(reduced_embs_ds_list)
			strategy_results['strategy'] = strategy.value
			self.results_collector.collect(current_configs, strategy_results)


	def _execute_plotting(self, strategy: ReductionStrategy, prot_embs_ds: Dataset, ster_embs_ds: Dataset) -> None:
		"""
		Executes the plotting step of the experiment.
		"""
		# Reduction based on PCA (trained on the protected embeddings)
		bidim_reducer: BaseDimensionalityReducer = TrainedPCAReducer(prot_embs_ds[EMB_COL], output_features=2)
		reduced_prot_embs_ds: Dataset = bidim_reducer.reduce_ds(prot_embs_ds)
		reduced_ster_embs_ds: Dataset = bidim_reducer.reduce_ds(ster_embs_ds)
		
		# Plotting the reduced embeddings
		plotter: DatasetPairScatterPlotter = DatasetPairScatterPlotter(reduced_prot_embs_ds, reduced_ster_embs_ds,
			title=f"Reduced embeddings for strategy: {strategy.name}")
		plotter.show()


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
		relevance_probabilities = DynamicPipelineExperiment._normalize_relevance(relevance)

		# For each feature, we sample it with a probability equal to its relevance
		reduced_embs[ReductionStrategy.RELEVANT_SAMPLING] = self._sample_with_probability(prot_embs_ds, ster_embs_ds, relevance_probabilities)

		# For each feature, we sample it with a probability equal to 1 - its relevance
		anti_relevance_probabilities = 1 - relevance_probabilities
		reduced_embs[ReductionStrategy.ANTI_RELEVANT_SAMPLING] = self._sample_with_probability(prot_embs_ds, ster_embs_ds, anti_relevance_probabilities)

		"""
		# unused

		# For each feature, we sample it with a probability equal to its relevance
		reduced_embs[ReductionStrategy.ENHANCED_RELEVANT_SAMPLING] = self._sample_with_probability(prot_embs_ds, ster_embs_ds, 
			DynamicPipelineExperiment._enhance_probabilities(relevance_probabilities))

		# For each feature, we sample it with a probability equal to 1 - its relevance
		anti_relevance_probabilities = 1 - relevance_probabilities
		reduced_embs[ReductionStrategy.ENHANCED_ANTI_RELEVANT_SAMPLING] = self._sample_with_probability(prot_embs_ds, ster_embs_ds,
			DynamicPipelineExperiment._enhance_probabilities(anti_relevance_probabilities))
		"""

		"""
		@deprecated
		
		# We choose the best value for N
		best_n: int = self._choose_best_n(prot_embs_ds, relevance)
		# We select the N features with the highest relevance
		top_n_relevant_features_reducer = WeightsSelectorReducer.from_weights(relevance, best_n)
		reduced_embs[ReductionStrategy.TOP_N_WITH_BEST_CHOICE] = top_n_relevant_features_reducer.reduce_ds(prot_embs_ds), top_n_relevant_features_reducer.reduce_ds(ster_embs_ds)

		# We select the N features with the lowest relevance
		anti_relevance = 1 / relevance
		anti_top_n_relevant_features_reducer = WeightsSelectorReducer.from_weights(anti_relevance, best_n)
		reduced_embs[ReductionStrategy.ANTI_TOP_N_WITH_BEST_CHOICE] = anti_top_n_relevant_features_reducer.reduce_ds(prot_embs_ds), anti_top_n_relevant_features_reducer.reduce_ds(ster_embs_ds)
		"""
		
		return reduced_embs


	@staticmethod
	def _normalize_relevance(relevance: torch.Tensor) -> torch.Tensor:
		"""
		Maps the relevance values to the range [0, 1].
		It assures that the minimum value is 0 and the maximum value is 1.

		:param relevance: The relevance values to normalize.
		:return: The normalized relevance values.
		"""
		rel_min = relevance.min()
		return (relevance - rel_min) / (relevance.max() - rel_min)


	@staticmethod
	def _enhance_probabilities(probabilities: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
		"""
		Enhances the probabilities by subtracting the mean and applying the sigmoid function.

		:param probabilities: The probabilities to enhance.
		:return: The enhanced probabilities.
		"""
		return torch.sigmoid(probabilities - threshold)


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