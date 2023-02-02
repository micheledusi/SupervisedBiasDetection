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
from itertools import product
import torch
from torchmetrics import PearsonCorrCoef

from tqdm import tqdm
from data_processing.sentence_maker import PP_PATTERN, SP_PATTERN
from experiments.base import Experiment
from experiments.midstep_analysis import MidstepAnalysisExperiment
from model.classification.base import AbstractClassifier
from model.reduction.composite import CompositeReducer
from model.reduction.pca import TrainedPCAReducer
from model.reduction.weights import WeightsSelectorReducer
from utils.cache import get_cached_embeddings, get_cached_mlm_scores


class MidstepAnalysis2Experiment(Experiment):

	def __init__(self) -> None:
		super().__init__("midstep analysis 2")

	def _get_property_embeddings(self, property: str, property_type: str, words_file_id: int, templates_file_id: int, **kwargs) -> Dataset:
		"""
		Returns the embeddings for the given property.

		:param property: The name of the property.
		:param property_type: The type of the property. Must be "protected" or "stereotyped".
		:param words_file_id: The id of the words file to use. (e.g. 01, 02, etc.)
		:param templates_file_id: The id of the templates file to use. (e.g. 01, 02, etc.)
		:param kwargs: Additional arguments to pass to the embedding function.
		:return: The embeddings for the given property.
		"""
		pattern = PP_PATTERN if property_type == 'protected' else SP_PATTERN if property_type == 'stereotyped' else None
		if pattern is None:
			raise ValueError(f'Invalid property type: {property_type}. Must be "protected" or "stereotyped".')
		words_file = f'data/{property_type}-p/{property}/words-{words_file_id:02d}.csv'
		templates_file = f'data/{property_type}-p/{property}/templates-{templates_file_id:02d}.csv'
		embeddings = get_cached_embeddings(property_name=property, property_pattern=pattern, words_file=words_file, templates_file=templates_file, rebuild=False, **kwargs)
		squeezed_embs = embeddings['embedding'].squeeze().tolist()
		embeddings = embeddings.remove_columns('embedding').add_column('embedding', squeezed_embs).with_format('pytorch')
		return embeddings

	def _get_embeddings(self, protected_property: str, stereotyped_property: str, num_max_tokens: int, num_templates: int) -> tuple[Dataset, Dataset]:
		"""
		Returns the embeddings for the protected and stereotyped property.

		:param protected_property: The name of the protected property.
		:param stereotyped_property: The name of the stereotyped property.
		:param num_max_tokens: The number of maximum tokens to consider in the embeddings.
		:param num_templates: The number of templates to consider in the embeddings.
		:return: A tuple containing the embeddings for the protected and stereotyped property.
		"""
		protected_embedding_dataset = self._get_property_embeddings(protected_property, 'protected', 1, 0, 
			max_tokens_number=num_max_tokens, 
			templates_selected_number=num_templates)
		stereotyped_embedding_dataset = self._get_property_embeddings(stereotyped_property, 'stereotyped', 1, 1, 
			max_tokens_number=num_max_tokens, 
			templates_selected_number=num_templates)
		return protected_embedding_dataset, stereotyped_embedding_dataset

	def _get_mlm_scores(self, protected_property: str, stereotyped_property: str, num_max_tokens: int) -> Dataset:
		"""
		Returns the MLM scores for the protected and stereotyped properties.

		:param protected_property: The name of the protected property.
		:param stereotyped_property: The name of the stereotyped property.
		:param num_max_tokens: The number of maximum tokens to consider in the embeddings.
		:return: The MLM scores dataset for the protected and stereotyped properties.
		"""
		generation_file_id: int = 1
		mlm_scores_ds = get_cached_mlm_scores(protected_property, stereotyped_property, generation_id=generation_file_id, max_tokens_number=num_max_tokens)
		mlm_scores_ds = mlm_scores_ds.rename_column('stereotyped_word', 'word')
		mlm_scores = mlm_scores_ds.sort('word')
		return mlm_scores
	
	def _check_correspondence(self, stereotyped_embedding_dataset: Dataset, mlm_scores: Dataset) -> None:
		assert len(stereotyped_embedding_dataset) == len(mlm_scores), f"Expected the same number of words in the \"stereotyped embeddings dataset\" and in the \"MLM scores dataset\", but got {len(stereotyped_embedding_dataset)} and {len(mlm_scores)}."
		for w1, w2 in zip(stereotyped_embedding_dataset['word'], mlm_scores['word']):
			assert w1 == w2, f"Expected the same words in the \"stereotyped embeddings dataset\" and in the \"MLM scores dataset\", but got {w1} and {w2}."

	def _reduce_with_midstep(self, prot_emb: Dataset, stere_emb: Dataset, midstep: int, classifier: AbstractClassifier) -> torch.Tensor:
		# Creating the reducer
		reducer_1 = WeightsSelectorReducer.from_classifier(classifier, output_features=midstep)
		reduced_protected_embeddings = reducer_1.reduce(prot_emb['embedding'])
		reducer_2: TrainedPCAReducer = TrainedPCAReducer(reduced_protected_embeddings, output_features=2)
		reducer = CompositeReducer([reducer_1, reducer_2])

		# Using the reducer to reduce the stereotyped embeddings
		return reducer.reduce(stere_emb['embedding'])

	def _compute_correlation(self, reduced_embeddings: torch.Tensor, mlm_scores: torch.Tensor) -> torch.Tensor:
		"""
		Computes the correlation (a similarity measure) between the reduced embeddings and the MLM scores.
		The result is a tensor where each element is the correlation between the MLM scores and a coordinate of the reduced embeddings.

		:param reduced_embeddings: The reduced embeddings
		:param mlm_scores: The MLM scores
		:return: A tensor where each element is the correlation between the MLM scores and a coordinate of the reduced embeddings. 
		E.g. if the embeddings are reduced to 2 dimensions, the result will be a tensor of size 2.
		"""
		assert reduced_embeddings.shape[0] == mlm_scores.shape[0], f"Expected the same number of embeddings and scores, but got {reduced_embeddings.shape[0]} and {mlm_scores.shape[0]}."
		# Computation
		coefs = []
		for polar_i in range(reduced_embeddings.shape[1]):
			emb_coord = reduced_embeddings.moveaxis(0, 1)[polar_i]
			pearson = PearsonCorrCoef()
			corr = pearson(emb_coord, mlm_scores)
			coefs.append(corr.item())
		return torch.Tensor(coefs)

	def _execute(self, **kwargs) -> None:
		protected_property = 'gender'
		stereotyped_property = 'profession'

		num_max_tokens = [1, 2]
		num_templates = [3]
		classifier_type = 'svm'

		results: Dataset = Dataset.from_dict({"n": list(range(2, 768))})

		for ntok, ntem in product(num_max_tokens, num_templates):
			print(f"Parameters:\n",
				f"\t- Number of maximum tokens: {ntok}\n",
				f"\t- Number of templates: {ntem}\n",
				f"\t- Classifier: {classifier_type}\n")
			# Getting the embeddings
			prot_emb, stere_emb = self._get_embeddings(protected_property, stereotyped_property, ntok, ntem)
			# Getting the MLM scores
			mlm_scores = self._get_mlm_scores(protected_property, stereotyped_property, ntok)
			self._check_correspondence(stere_emb, mlm_scores)
			# Identifying the score column
			for col in mlm_scores.column_names:
				if col.startswith('polarization'):
					score_column = col
					break
			# TODO: At the moment, this experiment does not support multiple score columns.
			# NOTE: Multiple score columns are generated when the classifier has more that two classes.

			# Creating and training the classifier
			classifier: AbstractClassifier = MidstepAnalysisExperiment._get_classifier(classifier_type)
			classifier.train(prot_emb)

			# Computing the correlations
			correlations = []
			for n in tqdm(range(2, 768)):
			
				# First, we compute the 2D embeddings with the composite reducer, with the current value of midstep "n"
				# Note: if an exception occurs, we stop the experiment
				try:
					reduced_embeddings = self._reduce_with_midstep(prot_emb, stere_emb, n, classifier)
					current_correlation = self._compute_correlation(reduced_embeddings, mlm_scores[score_column])
					# The resulting tensor has a correlation value for each coordinate of the reduced embeddings (in this case, 2)
					correlations.append(current_correlation)
				except RuntimeError as e:
					print(f"An exception occurred while reducing the embeddings with midstep {n}:\n{e}\nStopping the experiment.")
					break
			print()
			# Stack the correlations to get a tensor of shape (num_midsteps, 2)
			correlations: torch.Tensor = torch.stack(correlations)
			# Move the first axis to the second, to get a tensor of shape (2, num_midsteps)
			correlations = correlations.moveaxis(0, 1)
			for dim in range(correlations.shape[0]):
				# Add the correlation values to the results dataset
				results.add_column(f"TK{ntok}_TM{ntem}_DIM{dim}", correlations[dim].tolist())

		# Save the results
		folder = f"results/{protected_property}-{stereotyped_property}"
		if not os.path.exists(folder):
			os.makedirs(folder)
		filename = f"aggregated_midstep_correlation_CL{classifier_type}.csv"
		results.to_csv(f"{folder}/{filename}", index=False)