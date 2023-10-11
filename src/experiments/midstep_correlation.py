# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# Analyzing the role of hyperparameter "n" in dimensionality reduction.
# Compared to the previous version of this experiment, this one considers
# multiple embeddings for each word. That means that we can compare the 
# embeddings obtained with different hyperparameters (e.g. number of templates, number of maximum tokens, etc.)
# and see how they perform in terms of bias detection.

import os
from datasets import Dataset
import numpy as np
import torch
from torchmetrics import PearsonCorrCoef

from tqdm import tqdm
from data_processing.data_reference import BiasDataReference, PropertyDataReference
from experiments.base import Experiment, REBUILD
from model.classification.base import AbstractClassifier
from model.binary_scoring.polarization.base import PolarizationScorer
from model.classification.factory import ClassifierFactory
from model.reduction.composite import CompositeReducer
from model.reduction.pca import TrainedPCAReducer
from model.reduction.weights import WeightsSelectorReducer
from utils.caching.creation import get_cached_polarization_scores
from utils.config import Configurations, ConfigurationsGrid, Parameter
from utils.const import DEVICE


# Configurations to process data
configurations = ConfigurationsGrid({
	Parameter.MAX_TOKENS_NUMBER: 'all',
	Parameter.TEMPLATES_SELECTED_NUMBER: 'all',
	Parameter.CLASSIFIER_TYPE: 'svm',
	Parameter.CROSSING_STRATEGY: 'pppl',
	Parameter.POLARIZATION_STRATEGY: ['difference', 'ratio'],
})

BIAS_GENERATION_ID = 1
OUTPUT_NAME_MIDSTEP_ANALYSIS = "aggregated_midstep_correlation"
OUTPUT_NAME_REDUCED_EMBEDDINGS = "reduced_embeddings_correlation"


class MidstepAnalysisCorrelation(Experiment):

	def __init__(self) -> None:
		super().__init__("midstep analysis 2", required_kwargs=['prot_prop', 'ster_prop'])

	def _get_polarization_scores(self, configs: Configurations) -> Dataset:
		"""
		Returns the polarization cross-scores for the protected and stereotyped properties.
		The cross scores are computed accordijng to the given parameters.

		:param num_max_tokens: The number of maximum tokens to consider in the embeddings.
		:param cross_score_type: The type of cross-score to compute. Current supported values are 'pppl' and 'mlm'.
		:param polarization_strategy: The strategy to use to compute the polarization. Current supported values are 'difference' and 'ratio'.
		:return: The Cross-scores dataset for the protected and stereotyped properties. Each row is associated with a stereotyped value/word,
			while each column is associated with a polarization between two protected values. The values in the dataset are the polarizations between cross-scores.
			E.g. For properties "gender" and "occupation", we can take the row for "nurse" and the column for the "male-female" polarization.
		"""
		bias_reference = BiasDataReference(self.protected_property, self.stereotyped_property, BIAS_GENERATION_ID)
		pp_entries, sp_entries, polarization_scores = get_cached_polarization_scores(bias_reference, configs=configs, rebuild=REBUILD)
		# # DEBUG
		# print("Protected entries length:", len(pp_entries))
		# print(pp_entries)
		# print("Stereotyped entries length:", len(sp_entries))
		# print(sp_entries)
		# print("Polarization scores length:", len(polarization_scores))
		# print(polarization_scores.data)

		if not PolarizationScorer.STEREOTYPED_ENTRIES_COLUMN in polarization_scores.column_names:
			raise ValueError(f'Expected the column "stereotyped_value" in the cross-scores dataset, but it was not found.')
		return polarization_scores.rename_column(PolarizationScorer.STEREOTYPED_ENTRIES_COLUMN, 'word').sort('word')
	
	def _check_correspondence(self, stereotyped_embedding_dataset: Dataset, polarization_scores: Dataset) -> None:
		"""
		Checks that the words in the "stereotyped embeddings dataset" and in the "polarization cross-scores dataset" are the same.
		"""
		assert len(stereotyped_embedding_dataset) == len(polarization_scores), f"Expected the same number of words in the \"stereotyped embeddings dataset\" and in the \"polarization cross-scores dataset\", but got {len(stereotyped_embedding_dataset)} and {len(polarization_scores)}."
		for w1, w2 in zip(stereotyped_embedding_dataset['word'], polarization_scores['word']):
			assert w1 == w2, f"Expected the same words in the \"stereotyped embeddings dataset\" and in the \"polarization cross-scores dataset\", but got {w1} and {w2}."

	def _reduce_with_midstep(self, prot_emb: Dataset, stere_emb: Dataset, midstep: int, classifier: AbstractClassifier) -> torch.Tensor:
		# Creating the reducer
		reducer_1 = WeightsSelectorReducer.from_classifier(classifier, output_features=midstep)
		prot_input: torch.Tensor = prot_emb['embedding'].to(DEVICE)
		reduced_protected_embeddings = reducer_1.reduce(prot_input).to(DEVICE)
		reducer_2: TrainedPCAReducer = TrainedPCAReducer(reduced_protected_embeddings, output_features=2)
		reducer = CompositeReducer([reducer_1, reducer_2])

		# Using the reducer to reduce the stereotyped embeddings
		return reducer.reduce(stere_emb['embedding'])

	def _compute_correlation(self, reduced_embeddings: torch.Tensor, polarization_scores: torch.Tensor) -> torch.Tensor:
		"""
		Computes the correlation (a similarity measure) between the reduced embeddings and the MLM scores.
		The result is a tensor where each element is the correlation between the MLM scores and a coordinate of the reduced embeddings.

		:param reduced_embeddings: The reduced embeddings
		:param polarization_scores: The MLM scores
		:return: A tensor where each element is the correlation between the MLM scores and a coordinate of the reduced embeddings. 
		E.g. if the embeddings are reduced to 2 dimensions, the result will be a tensor of size 2.
		"""
		assert reduced_embeddings.shape[0] == polarization_scores.shape[0], f"Expected the same number of embeddings and scores, but got {reduced_embeddings.shape[0]} and {polarization_scores.shape[0]}."
		#print("CORRELATION COMPUTING")
		#print("Embeddings shape:", reduced_embeddings.shape)
		#print("Scores shape:", polarization_scores.shape)
		# Computation
		coefs = []
		polarization_scores = polarization_scores.to(DEVICE)
		pearson = PearsonCorrCoef().to(DEVICE)
		for polar_i in range(reduced_embeddings.shape[1]):
			emb_coord = reduced_embeddings.moveaxis(0, 1)[polar_i].to(DEVICE)
			corr = pearson(emb_coord, polarization_scores)
			coefs.append(corr.item())
		return torch.Tensor(coefs).to(DEVICE)

	def _execute(self, **kwargs) -> None:

		results: Dataset = Dataset.from_dict({"n": list(range(2, 768+1))})

		last_configs: Configurations = None
		for configs in configurations:
			# Showing the current configuration
			print("Current parameters configuration:\n", configs, '\n')
			
			# Getting the embeddings
			prot_dataset, ster_dataset = self._get_embeddings(configs)
			# Getting the PPPL / MLM scores
			cross_scores = self._get_polarization_scores(configs)
			self._check_correspondence(ster_dataset, cross_scores)

			# Creating and training the classifier
			classifier: AbstractClassifier = ClassifierFactory.create(configs)
			classifier.train(prot_dataset)

			# For each polarization column, we compute the correlations
			polarization_prefix = 'polarization_'
			for pol_column in cross_scores.column_names:
				# Check if the column is a polarization column
				if pol_column.startswith(polarization_prefix):
					pol_name = pol_column[len(polarization_prefix):]
				else:
					continue

				# For each column of polarization scores, we compute the correlations
				# Computing the correlations
				correlations: list = []
				for n in tqdm(range(2, 768+1)):
				
					# First, we compute the 2D embeddings with the composite reducer, with the current value of midstep "n"
					# Note: if an exception occurs, we stop the experiment
					try:
						reduced_embeddings = self._reduce_with_midstep(prot_dataset, ster_dataset, n, classifier)
						current_correlation = self._compute_correlation(reduced_embeddings, cross_scores[pol_column].to(DEVICE))
						# The resulting tensor has a correlation value for each coordinate of the reduced embeddings (in this case, 2)
						correlations.append(current_correlation)
					except RuntimeError as e:
						print(f"An exception occurred while reducing the embeddings with midstep {n}:\n{e}\nStopping the experiment.")
						break

				# Stack the correlations to get a tensor of shape (num_midsteps, 2)
				correlations: torch.Tensor = torch.stack(correlations)
				# Move the first axis to the second, to get a tensor of shape (2, num_midsteps)
				correlations = correlations.moveaxis(0, 1)

				for dim in range(correlations.shape[0]):
					# Add the correlation values to the results dataset
					# Each column name will contain the MUTABLE parameters in the configuration, i.e.
					# the parameters that can change from one experiment to another.
					# The column name will also contain the name of the polarization column. 
					column_name: str = f"{configs.subget_mutables().to_abbrstr()}_{pol_name}_DIM{dim}"
					results = results.add_column(column_name, correlations[dim].tolist())
			
			last_configs = configs

		# Save the results
		folder: str = self._get_results_folder(last_configs, prot_dataset, ster_dataset)
		# The filename will contain the IMMUTABLE parameters in the configuration, i.e.
		# the parameters that cannot change from one experiment to another.
		filename = f"{OUTPUT_NAME_MIDSTEP_ANALYSIS}_{configs.subget_immutables().to_abbrstr()}.csv"
		results.to_csv(f"{folder}/{filename}", index=False)
		print(f"Results saved in {folder}/{filename}")

		###################################################################
		# Drawing

		# We take a specific configuration to draw the graphs
		draw_configs: Configurations = last_configs
		corr_max = 0
		pol_axis_max = ""
		n_max = 0
		for col in results.column_names:
			# Check if the column is a polarization column
			if not col.startswith(draw_configs.subget_mutables().to_abbrstr()):
				continue
			# Getting the max correlation for each dimension
			column_correlations = torch.Tensor([abs(x) for x in results[col]])
			# Getting the index of the max correlation
			n = column_correlations.argmax().item()
			if column_correlations[n] > corr_max:
				corr_max = column_correlations[n]
				pol_axis_max = col.split("_")[-3:-1]
				n_max = n
		first_prot_value, second_prot_value = pol_axis_max
		
		# Print data
		print(f"Max correlation: {corr_max}")
		print(f"Found in column: {pol_axis_max}, with midstep n = {n_max}")

		# Now we have the column with the max correlation
		prot_dataset, ster_dataset = self._get_embeddings(draw_configs)
		cross_scores = self._get_polarization_scores(draw_configs)
		# Extracting the polarization scores for the Stereotyped words
		sp_polar = cross_scores[f"polarization_{first_prot_value}_{second_prot_value}"].tolist()
		first_prot_value_polar = min(sp_polar)
		second_prot_value_polar = max(sp_polar)

		""" OLD VERSION
		pp_polar = list(map(lambda val: 
		 	first_prot_value_polar if val == first_prot_value 
		 	else second_prot_value_polar if val == second_prot_value
			else None,  
			prot_dataset['value'])) """
		pp_polar = [np.mean(sp_polar)] * len(prot_dataset['value'])
		
		polarizations = sp_polar + pp_polar
		types = ['stereotyped'] * len(sp_polar) + ['protected'] * len(pp_polar)
		words = ster_dataset['word'] + prot_dataset['word']
		values = ster_dataset['value'] + prot_dataset['value']

		# Creating and training the classifier
		classifier: AbstractClassifier = ClassifierFactory.create(draw_configs)
		classifier.train(prot_dataset)
		
		reducer_1 = WeightsSelectorReducer.from_classifier(classifier, output_features=n_max)
		prot_input: torch.Tensor = prot_dataset['embedding'].to(DEVICE)
		reduced_protected_embeddings = reducer_1.reduce(prot_input).to(DEVICE)
		reducer_2: TrainedPCAReducer = TrainedPCAReducer(reduced_protected_embeddings, output_features=2)
		reducer = CompositeReducer([reducer_1, reducer_2])

		# Using the reducer to reduce BOTH protected and stereotyped embeddings
		all_embeddings = torch.cat((ster_dataset['embedding'], prot_dataset['embedding']))
		reduced_embeddings = reducer.reduce(all_embeddings).to(DEVICE)
		print("Reduced embeddings shape:", reduced_embeddings.shape)

		first_coord = reduced_embeddings[:, 0].tolist()
		second_coord = reduced_embeddings[:, 1].tolist()

		plot_results = Dataset.from_dict({
			"word": words,
			"value": values,
			'type': types,
			"x": first_coord, 
			"y": second_coord,
			"polarization": polarizations
			})
	
		# Print the reduced embeddings to CSV file
		config_str = draw_configs.to_abbrstr()
		filename = f"{OUTPUT_NAME_REDUCED_EMBEDDINGS}_{config_str}_n{n_max}.csv"
		plot_results.to_csv(f"{folder}/{filename}", index=False)
		print(f"Reduced embeddings saved in {folder}/{filename}")


