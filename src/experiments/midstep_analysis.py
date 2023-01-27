# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# Analyzing the role of hyperparameter "n" in dimensionality reduction


import os
import torch
from torchmetrics import PearsonCorrCoef
from datasets import Dataset
from tqdm import tqdm
from experiments.base import Experiment
from model.reduction.composite import CompositeReducer
from model.reduction.pca import TrainedPCAReducer
from model.reduction.weights import WeightsSelectorReducer
from model.regression.linear_regressor import LinearRegressor

"""
from datasets import disable_caching
disable_caching()"""


class MidstepAnalysisExperiment(Experiment):

	def __init__(self) -> None:
		super().__init__("midstep analysis")

	def _execute(self, **kwargs) -> None:
		protected_property = 'gender'
		stereotyped_property = 'profession'

		# Getting embeddings (as Dataset objects)
		protected_embedding_dataset, stereotyped_embedding_dataset = Experiment._get_default_embeddings(protected_property, stereotyped_property)
		stereotyped_embedding_dataset = stereotyped_embedding_dataset.sort('word')

		# Getting MLM scores (as Dataset object)
		mlm_scores_ds = Experiment._get_default_mlm_scores(protected_property, stereotyped_property)
		mlm_scores_ds = mlm_scores_ds.rename_column('stereotyped_word', 'word')
		mlm_scores = mlm_scores_ds.sort('word')

		# Checking that the two datasets have the same words
		assert len(stereotyped_embedding_dataset) == len(mlm_scores), f"Expected the same number of words in the \"stereotyped embeddings dataset\" and in the \"MLM scores dataset\", but got {len(stereotyped_embedding_dataset)} and {len(mlm_scores)}."
		for w1, w2 in zip(stereotyped_embedding_dataset['word'], mlm_scores['word']):
			assert w1 == w2, f"Expected the same words in the \"stereotyped embeddings dataset\" and in the \"MLM scores dataset\", but got {w1} and {w2}."

		# TODO: this is going to be removed
		num_polarizations = 0
		for column in mlm_scores.column_names:
			if column.startswith('polarization'):
				score_column = column
				num_polarizations += 1
		if num_polarizations > 1:
			raise NotImplementedError('The dataset "mlm_scores" contains more than one column representing a "polarization axis". This is not currently supported by this experiment.')
		# TODO END

		# For every value of n (called 'midstep'), run the experiment
		scores: dict = {'midstep': [], 'correlation': []}
		for midstep in tqdm(range(2, 768, 24)):
			# First, we compute the 2D embeddings with the composite reducer, with the current value of midstep "n"
			reduced_embeddings = MidstepAnalysisExperiment._reduce_with_midstep(protected_embedding_dataset, stereotyped_embedding_dataset, midstep)

			# Then, we compare the reduced embeddings with the mlm scores
			correlation = MidstepAnalysisExperiment._compute_correlation(reduced_embeddings, mlm_scores[score_column])
			scores['midstep'].append(midstep)
			scores['correlation'].append(correlation)

			if midstep >= 50:
				# Saving the embeddings
				reduced_embeddings = mlm_scores.add_column('x', reduced_embeddings[:, 0].tolist()).add_column('y', reduced_embeddings[:, 1].tolist())
				reduced_embeddings.to_csv(f"results/{protected_property}-{stereotyped_property}/reduced_embeddings_N{midstep}.csv", index=False)
				exit()
		
		scores: Dataset = Dataset.from_dict(scores)
		
		def prepare_fn(sample):
			sample['x-correlation'] = abs(sample['correlation'][0])
			sample['y-correlation'] = abs(sample['correlation'][1])
			return sample
		scores = scores.map(prepare_fn, batched=False, remove_columns=['correlation']).rename_column('midstep', 'n')

		# Finally, we save the results
		folder = f"results/{protected_property}-{stereotyped_property}"
		if not os.path.exists(folder):
			os.makedirs(folder)
		scores.to_csv(f"{folder}/midstep_correlation.csv", index=False)

	
	@staticmethod
	def _get_composite_reducer(prot_emb_ds: Dataset, midstep: int) -> CompositeReducer:
		"""
		Buils a composite reducer that first reduces the embeddings using the weights of the classifier and then
		reduces the result using PCA.
		The number of features in the first reduction is given by the parameter 'midstep'. At the end, the number of features is 2.

		:param prot_emb_ds: The dataset containing the protected embeddings
		:param midstep: The number of features to use in the first reduction
		"""
		regressor: LinearRegressor = LinearRegressor()
		regressor.train(prot_emb_ds)
		reducer_1 = WeightsSelectorReducer.from_regressor(regressor, output_features=midstep)
		reduced_protected_embeddings = reducer_1.reduce(prot_emb_ds['embedding'])
		reducer_2: TrainedPCAReducer = TrainedPCAReducer(reduced_protected_embeddings, output_features=2)
		reducer = CompositeReducer([reducer_1, reducer_2])
		return reducer
	
	@staticmethod
	def _reduce_with_midstep(prot_emb_ds: Dataset, ster_emb_ds: Dataset, midstep: int) -> torch.Tensor:
		reducer: CompositeReducer = MidstepAnalysisExperiment._get_composite_reducer(prot_emb_ds, midstep)
		return reducer.reduce(ster_emb_ds['embedding'])

	@staticmethod
	def _compute_correlation(reduced_embeddings: torch.Tensor, mlm_scores: torch.Tensor) -> torch.Tensor:
		"""
		Computes the correlation (a similarity measure) between the original embeddings and the reduced embeddings.
		The result is a tensor where each element is the correlation between the MLM scores and a coordinate of the reduced embeddings.

		:param reduced_embeddings: The reduced embeddings
		:param mlm_scores: The MLM scores
		"""
		assert reduced_embeddings.shape[0] == mlm_scores.shape[0], f"Expected the same number of embeddings and scores, but got {reduced_embeddings.shape[0]} and {mlm_scores.shape[0]}."
		
		print("Reduced embeddings shape:", reduced_embeddings.shape)
		print("MLM scores shape:", mlm_scores.shape)

		coefs = []
		for polar_i in range(reduced_embeddings.shape[1]):
			emb_coord = reduced_embeddings[:, polar_i]

			pearson = PearsonCorrCoef()
			corr = pearson(emb_coord, mlm_scores)
			coefs.append(corr.item())
		
		return torch.Tensor(coefs)




