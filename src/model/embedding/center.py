# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module contains the class "EmbeddingCenterer" which is used to center the embeddings.

import logging
from datasets import Dataset
import torch

from utils.config import Configurable, Configurations, Parameter


class EmbeddingCenterer(Configurable):
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
		super(EmbeddingCenterer, self).__init__(configs, parameters=[
            Parameter.CENTER_EMBEDDINGS,
			])

	def center(self, dataset: Dataset) -> Dataset:
		"""
		This method centers the embeddings, only if the CENTER_EMBEDDINGS parameter is set to True.

		NOTE: If the centering is performed, the method creates a copy of the dataset and does not modify the original one.
		Otherwise, it returns the original dataset.

		:param dataset: The dataset containing the embeddings.
		:return: The dataset with the embeddings centered.
		"""
		if not self.configs[Parameter.CENTER_EMBEDDINGS]:
			logging.warning("The embeddings are not being centered, because the CENTER_EMBEDDINGS parameter is set to False. Returning the original dataset.")
			return dataset

		# Getting embeddings
		if "embedding" not in dataset.column_names:
			error_msg: str = "The dataset does not contain the embeddings in the 'embedding' column. Please provide a dataset containing the embeddings."
			logging.error(error_msg)
			raise ValueError(error_msg)
		embs: torch.Tensor = dataset["embedding"]
		logging.debug("Embeddings shape: %s", str(embs.shape))

		# Computing mean
		mean: torch.Tensor = torch.mean(embs, dim=0)

		# Subtracting mean
		centered_embs: torch.Tensor = embs - mean

		# Creating new dataset with centered embeddings
		centered_dataset: Dataset = dataset \
			.remove_columns("embedding") \
			.add_column("embedding", centered_embs.tolist()) \
			.with_format("torch")
		return centered_dataset