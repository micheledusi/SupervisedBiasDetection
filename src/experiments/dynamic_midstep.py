# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2024				#
# - - - - - - - - - - - - - - - #

# This module is responsible for a general experiment.
# In this experiment, we try several strategies to choose the best value of the midstep a priori.

import logging
from colorist import Color
from datasets import Dataset, concatenate_datasets
import torch
from tqdm import tqdm

from experiments.base import Experiment
from model.classification.base import AbstractClassifier
from model.classification.chi import ChiSquaredTest, FisherCombinedProbabilityTest, HarmonicMeanPValue, ContingencyTable
from model.classification.factory import ClassifierFactory
from model.embedding.center import EmbeddingCenterer
from model.embedding.cluster_validator import ClusteringScorer
from model.embedding.combinator import EmbeddingsCombinator
from model.reduction.base import BaseDimensionalityReducer
from model.reduction.factory import ReductionFactory
from model.reduction.pca import JointPCAReducer, TrainedPCAReducer
from model.reduction.weights import WeightsSelectorReducer
from model.relevance.base import BaseRelevanceCalculator
from model.relevance.factory import RelevanceCalculatorFactory
from model.relevance.normalize import RelevanceNormalizer
from utils.caching.creation import get_cached_raw_embeddings
from utils.config import Configurations, Parameter
from view.plotter.scatter import DatasetPairScatterPlotter
from view.plotter.weights import WeightsPlotter

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


REBUILD_DATASETS = False

DO_PLOT = False
DO_SHOW_PLOT = DO_PLOT and False
DO_SAVE_PLOT = DO_PLOT and False
DO_PLOT_FOR_LATEX = DO_PLOT and False

DO_CHI2 = True
DO_PRINT_CONTINGENCY_TABLE = True


class DynamicPipelineExperiment(Experiment):
	"""
	In this experiment, we try several strategies to choose which dimensionality reduction suits best the bias detection method.
	"""

	DatasetPair = tuple[Dataset, Dataset]

	def __init__(self, configs: Configurations) -> None:
		super().__init__(
			"dynamic_pipeline", 
			required_kwargs=['prot_prop', 'ster_prop'], 
			configs=configs
			)
		self.weights_already_plotted = False


	def _execute(self, **kwargs) -> None:	
		# For every combination of parameters (for the step of the raw embeddings computation)
		for configs_re in self.configs.iterate_over(Configurations.ParametersSelection.RAW_EMBEDDINGS_COMPUTATION):
			
			# Loading the datasets
			protected_property_ds: Dataset = get_cached_raw_embeddings(self.protected_property, configs_re, REBUILD_DATASETS)
			stereotyped_property_ds: Dataset = get_cached_raw_embeddings(self.stereotyped_property, configs_re, REBUILD_DATASETS)

			logger.debug(f"Resulting protected raw dataset for property \"{self.protected_property.name}\":\n{protected_property_ds}")
			logger.debug(f"Resulting stereotyped raw dataset for property \"{self.stereotyped_property.name}\":\n{stereotyped_property_ds}")
		
			# Combining the embeddings
			combinator = EmbeddingsCombinator(configs_re)

			combined_protected_embeddings: dict[Configurations, list[Dataset]] = combinator.combine(protected_property_ds)
			combined_stereotyped_embeddings: dict[Configurations, list[Dataset]] = combinator.combine(stereotyped_property_ds)

			# Now we have the combined embeddings, we can proceed with the preprocessing and the bias detection
			for key_configs in combined_protected_embeddings:

				# We assume that the keys of the two dictionaries are the same
				# Meaning that the configurations are the same for both the protected and the stereotyped property
				protected_embs_ds_list: list[Dataset] = combined_protected_embeddings[key_configs]
				stereotyped_embs_ds_list: list[Dataset] = combined_stereotyped_embeddings[key_configs]

				for curr_preproc_configs in key_configs.iterate_over([Parameter.CENTER_EMBEDDINGS, Parameter.PREREDUCE_EMBEDDINGS]):

					# Centering the embeddings
					if curr_preproc_configs[Parameter.CENTER_EMBEDDINGS]:
						centerer: EmbeddingCenterer = EmbeddingCenterer(curr_preproc_configs)
						protected_embs_ds_list: list[Dataset] = [centerer.center(ds) for ds in protected_embs_ds_list]
						stereotyped_embs_ds_list: list[Dataset] = [centerer.center(ds) for ds in stereotyped_embs_ds_list]
					
					# Pre-reducting the embeddings with PCA
					if curr_preproc_configs[Parameter.PREREDUCE_EMBEDDINGS]:
						# We iterate over the number of dimensions to reduce the embeddings to
						for curr_preproc_bydim_configs in curr_preproc_configs.iterate_over([Parameter.PREREDUCTION_DIMENSIONS]):
							# Now we reduce by this number of dimensions
							prereduce_dims: int = curr_preproc_bydim_configs[Parameter.PREREDUCTION_DIMENSIONS]
							pca_prereducer = JointPCAReducer(output_features=prereduce_dims)
							new_protected_embs_ds_list: list[Dataset] = []
							new_stereotyped_embs_ds_list: list[Dataset] = []
							for prot_ds, ster_ds in zip(protected_embs_ds_list, stereotyped_embs_ds_list):
								prot_ds, ster_ds = pca_prereducer.reduce_both_ds(prot_ds, ster_ds)
								new_protected_embs_ds_list.append(prot_ds)
								new_stereotyped_embs_ds_list.append(ster_ds)

							# We run the experiment for each configuration of the embeddings
							logger.info(f"Running experiment for configuration:\n{curr_preproc_bydim_configs}")
							self._execute_for_single_configuration(curr_preproc_bydim_configs, new_protected_embs_ds_list, new_stereotyped_embs_ds_list)
					else:
						# We run the experiment for each configuration of the embeddings
						logger.info(f"Running experiment for configuration:\n{curr_preproc_configs}")
						self._execute_for_single_configuration(curr_preproc_configs, protected_embs_ds_list, stereotyped_embs_ds_list)


	def _execute_for_single_configuration(self, configs: Configurations, prot_embs_ds_list: list[Dataset], ster_embs_ds_list: list[Dataset]) -> None:
		"""
		Executes the experiment for a single configuration of the embeddings.

		:param prot_embs_ds_list: The list of datasets of the protected embeddings.
		:param ster_embs_ds_list: The list of datasets of the stereotyped embeddings.
		"""
		# Phase 1: the reduction step
		reduced_embs_ds_pair_list_by_configs: dict[Configurations, list[tuple[Dataset, Dataset]]] = {}
		reducer_factory = ReductionFactory(configs)

		# This cycle is executed for each testcase
		for prot_embs_ds, ster_embs_ds in tqdm(zip(prot_embs_ds_list, ster_embs_ds_list), desc="Reduction step", total=len(prot_embs_ds_list)):
   
			# Phase 1.2: the reduction step with the selected strategy
			# We retrieve the list of reducers to use for the reduction step
			# The reducers are created with the configurations of the experiment, by the factory "ReductionFactory"
			for reducer_config, reducer in reducer_factory.create_multiple().items():
				if reducer.requires_training:
					reducer.train_ds(prot_embs_ds)
				reduced_prot_embs_ds, reduced_ster_embs_ds = reducer.reduce_both_ds(prot_embs_ds, ster_embs_ds)
				# We save the reduced embeddings for each testcase
				if reducer_config not in reduced_embs_ds_pair_list_by_configs:
					reduced_embs_ds_pair_list_by_configs[reducer_config] = []
				reduced_embs_ds_pair_list_by_configs[reducer_config].append((reduced_prot_embs_ds, reduced_ster_embs_ds))

		# DEBUG
		logger.debug(f"After the reductions, we obtained:")
		for key, value in reduced_embs_ds_pair_list_by_configs.items():
			logger.debug(f"Configuration: {key}")
			for prot_embs_ds, ster_embs_ds in value:
				logger.debug(f"Reduced protected embeddings:   {prot_embs_ds}")
				logger.debug(f"Reduced stereotyped embeddings: {ster_embs_ds}")

		# Phase 1.5: plotting the reduced embeddings with the ScatterPlotter
		if DO_PLOT:
			# For each strategy
			for reducer_config, reduced_embs_ds_pair_list in reduced_embs_ds_pair_list_by_configs.items():
				# For each testcase in this strategy
				for (prot_embs_ds, ster_embs_ds) in reduced_embs_ds_pair_list:
					self._execute_plotting(reducer_config.to_abbrstr(), prot_embs_ds, ster_embs_ds)
					break	# We plot only the first testcase
		
		# Phase 2: crossing the protected embeddings with the stereotyped embeddings to measure the bias
		if DO_CHI2:
			# For each configuration/reducer
			for reducer_config, reduced_embs_ds_pair_list in reduced_embs_ds_pair_list_by_configs.items():
				# If we have multiple strategies for the bias evaluation, we need to execute the experiment for each of them
				for bias_eval_config in reducer_config.iterate_over(Configurations.ParametersSelection.BIAS_EVALUTATION):

					strategy_str: str = reducer_factory.get_strategy_str(bias_eval_config)
					logger.info(f"Computing Chi-Squared value for STRATEGY: {Color.GREEN}{strategy_str}{Color.OFF}")
					# print(f"Configurations: {bias_eval_config}")

					# Computing the bias evaluation
					strategy_results: dict = self._execute_crossing(bias_eval_config, reduced_embs_ds_pair_list)

					# Printing the results
					print(f"Results for REDUCTION STRATEGY: {Color.GREEN}{strategy_str}{Color.OFF}")
					for key, value in strategy_results.items():
						str_value = f"{Color.YELLOW}{value:.10e}{Color.OFF}" if key == "harmonic_mean_p_value" else value
						print(f"{Color.CYAN}{key:<22s}{Color.OFF}: {str_value}")

					# Saving the results
					strategy_results["strategy_description"] = strategy_str
					self.results_collector.collect(bias_eval_config.expand_by(configs), strategy_results, remove_list_parameters=True)



	def _execute_plotting(self, strategy_str: str, prot_embs_ds: Dataset, ster_embs_ds: Dataset) -> None:
		"""
		Executes the plotting step of the experiment.
		"""
		# Reduction based on PCA (trained on the protected embeddings)
		bidim_reducer: BaseDimensionalityReducer = TrainedPCAReducer(output_features=2)
		reduced_prot_embs_ds: Dataset = bidim_reducer.reduce_ds(prot_embs_ds)
		reduced_ster_embs_ds: Dataset = bidim_reducer.reduce_ds(ster_embs_ds)
		
		# Plotting the reduced embeddings
		plotter: DatasetPairScatterPlotter = DatasetPairScatterPlotter(reduced_prot_embs_ds, reduced_ster_embs_ds,
			title=f"Reduced embeddings for strategy: {strategy_str}", use_latex=DO_PLOT_FOR_LATEX)
		# plotter.show()
		extension: str = "pdf" if DO_PLOT_FOR_LATEX else "png"
		img_filename = self._get_results_folder(self.configs) + f"/plot_scatter_{strategy_str}.{extension}"
		if DO_SHOW_PLOT:
			plotter.show()
		if DO_SAVE_PLOT:
			plotter.save(img_filename)


	def _execute_crossing(self, config: Configurations, embs_ds_list: list[tuple[Dataset, Dataset]]) -> dict[str, float]:
		"""
		Compares the protected embeddings with the stereotyped embeddings to measure the bias.
		The comparison is done using the chi-squared test for each testcase.
		Then, the results are combined using the Fisher's method and the Harmonic mean.

		:param embs_ds_list: The list of datasets of the embeddings: each element is a tuple containing the protected and the stereotyped embeddings datasets.
		:return: A dictionary containing the results of the comparison.
		"""
		chi2 = ChiSquaredTest()
		contingency_tables_list: list[ContingencyTable] = []
		chi2_values_list: list[tuple[float, float]] = []

		# For each dataset/testcase of protected embeddings and the stereotyped embeddings
		for prot_dataset, ster_dataset in embs_ds_list:

			# Creating and training the reduction classifier
			factory: ClassifierFactory = ClassifierFactory(config, phase=ClassifierFactory.PHASE_CROSS)
			classifier: AbstractClassifier = factory.create()
			classifier.train(prot_dataset)

			# We evaluate the predictions of the classifier on the stereotyped embeddings
			ster_dataset_with_predictions: Dataset = classifier.prediction_to_value(classifier.evaluate(ster_dataset))

			# Chi-Squared test
			chi2_results = chi2.test_from_dataset(ster_dataset_with_predictions, "value", "predicted_value")
			# we obtain the chi-squared value and the p-value for this testcase
			chi2_values_list.append(chi2_results)
			contingency_tables_list.append(chi2.contingency_table)
		
		# Computing aggregated results
		chi2_values_list: torch.Tensor = torch.tensor(chi2_values_list)
		chi2_averages = chi2_values_list.mean(dim=0)
		chi2_std_devs = chi2_values_list.std(dim=0)

		logger.debug(f"List of Chi-Squared values: {chi2_values_list}")
		logger.debug(f"Averages of the previous list: {chi2_averages}")
		
		"""
		# # Saving the testcases results along with the number of dimensions
		# num_features_list: list[int] = [prot_ds[COL_EMBS].shape[-1] for prot_ds, _ in embs_ds_list]
		# testcase_results: Dataset = Dataset.from_dict({
		# 	"strategy": [strategy_str] * len(embs_ds_list),
		# 	"num_features": num_features_list,
		# 	"chi2": chi2_values_list.select(dim=1, index=0),
		# 	"p_value": chi2_values_list.select(dim=1, index=1),
		# })
		# if not self.testcase_results_ds:
		# 	self.testcase_results_ds = testcase_results
		# else:
		# 	self.testcase_results_ds = concatenate_datasets([self.testcase_results_ds, testcase_results])
		"""
		# Averaging the contingency tables
		if DO_PRINT_CONTINGENCY_TABLE:
			strategy_str = ReductionFactory(config).get_strategy_str(config)
			logger.debug(f"Averaging the contingency tables for strategy {strategy_str} over {len(contingency_tables_list)} testcases...")
			aggregated_contingency_table = ChiSquaredTest.average_contingency_matrices(contingency_tables_list)
			aggregated_table_chi2_results = chi2.test_from_contingency_table(aggregated_contingency_table)

			print(chi2.get_formatted_table("AGGREGATED:"))

			# Saving the aggregated contingency table as LaTeX
			with open(self._get_results_folder(config) + "/contingency_tables.tex", "a") as f:
				f.write(f"Aggregated contingency table for strategy \\emph{{{strategy_str}}}:\n\n")
				f.write(chi2.get_formatted_table(f"STRATEGY:", use_latex=True))
				f.write("\n\n")
		

		"""
		# Combining the results with the Fisher's method
		fisher_test = FisherCombinedProbabilityTest()
		x2, degs, p_value = fisher_test.test(chi2_values_list.select(dim=1, index=1))	# We only need the p-values
		"""

		# Combining the results with the Harmonic mean
		harmonic_mean = HarmonicMeanPValue()
		harmonic_mean_p_value = harmonic_mean.test(chi2_values_list.select(dim=1, index=1))	# We only need the p-values

		"""
		# Printing the results
		logger.info("Configurations for the experiment:\n%s", config)
		# print(f"{Effect.BOLD}AGGREGATED RESULTS{Effect.OFF}:     AVG ± STD		over {len(embs_ds_list)} testcases")
		# print(f"Chi-Squared value:   {chi2_averages[0]:6.3f} ± {chi2_std_devs[0]:5.3f}")
		# print(f"{Effect.BOLD}COMBINED RESULTS{Effect.OFF}:")
		# print(f"Fisher Test value:   {x2:.3f} with {degs} degrees of freedom")
		# print(f"p-value:             {p_value}")
		# print(f"Harmonic mean p-value: {harmonic_mean_p_value}")
		"""

		return dict({
			"chi2_avg": chi2_averages[0],
			"chi2_std_dev": chi2_std_devs[0],
			# "fisher_x2": x2,
			# "fisher_degrees": degs,
			# "fisher_p_value": p_value,
			"harmonic_mean_p_value": harmonic_mean_p_value,
			# "aggregated_table_chi2": aggregated_table_chi2_results[0],
			# "aggregated_table_p_value": aggregated_table_chi2_results[1],
		})