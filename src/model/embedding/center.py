# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module contains the class "EmbeddingCenterer" which is used to center the embeddings.

from datasets import Dataset
import torch

from utils.config import Configurations


class EmbeddingCenterer:
	"""
	This class is used to center the embeddings.
	Its main method, "center", returns the embeddings centered around the mean.
	It takes as input a dataset containing the embeddings, in the "embedding" column.
	It computes the mean of the embeddings and then subtracts it from each embedding.
	"""
	def __init__(self, configs: Configurations) -> None:
		"""
		The initializer for the EmbeddingCenterer class.
		"""
		# NOTE: "configs" is not used in this class.
		# It is only passed to the initializer for future use.
		pass

	def center(self, dataset: Dataset) -> Dataset:
		"""
		This method centers the embeddings.
		NOTE: It creates a copy of the dataset and does not modify the original one.

		:param dataset: The dataset containing the embeddings.
		:return: The dataset with the embeddings centered.
		"""
		# Getting embeddings
		if "embedding" not in dataset.column_names:
			raise ValueError("The dataset does not contain the embeddings in the 'embedding' column. Please provide a dataset containing the embeddings.")
		embs: torch.Tensor = dataset["embedding"]
		print("Embeddings shape: ", embs.shape)

		# Computing mean
		mean: torch.Tensor = torch.mean(embs, dim=0)

		# Subtracting mean
		centered_embs: torch.Tensor = embs - mean

		# Creating new dataset with centered embeddings
		centered_dataset: Dataset = dataset.\
			remove_columns("embedding").\
			add_column("embedding", centered_embs.tolist()).\
			with_format("torch")
		return centered_dataset