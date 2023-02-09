# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

import torch
from transformers import AutoModelForMaskedLM

from model.cross_scoring.base import CrossScorer
from utils.const import DEFAULT_BERT_MODEL_NAME


class MLMCrossScorer(CrossScorer):
	"""
	This class implements the CrossScorer interface for Masked Language Models.
	The measure it computes is the probability of the target protected word, given a context containing the stereotyped word.
	E.g.:
		- Protected word: "she"  (value "female" for the <gender> protected property)
		- Stereotyped word: "nurse" (value "nurse" for the <profession> stereotyped property)
		- Context: "she is a nurse."
	The output of the model will be the probability of the word "she" given the context "[x] is a nurse.", 
	computed by the masked language model (MLM).

	A high score means that the model is more likely to predict the protected word in the context of the stereotyped word.
	I.e. the protected word is more likely to be associated with the stereotyped word.
	We cannot say anything about the opposite direction, i.e. whether the stereotyped word is more likely to be associated with the protected word.

	For this, this cross-scoring method is not symmetric: the order of the protected and stereotyped words matters.
	"""

	__softmax = torch.nn.Softmax(dim=-1)

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		# Getting the model
		self.model = AutoModelForMaskedLM.from_pretrained(DEFAULT_BERT_MODEL_NAME)
		if torch.cuda.is_available():
			self.model.cuda()

	def _get_subsequence_index(self, array: torch.Tensor, subarray: torch.Tensor) -> torch.Tensor:
		# Example:
		# array = [1, 2, 3, 9, 5, 6, 7, 8, 3, 4, 5, 10]
		# subarray = [3, 4, 5]
		# output = [8]
		window_len = subarray.shape[-1]
		steps = array.shape[-1] - window_len + 1
		# Unfold the last dimension of the array into 2 dimension of length [len(array) - window_len + 1, window_len]
		unfolded_array = array.unfold(dimension=-1, size=window_len, step=1)
		# Repeat the subarray to match the shape of the unfolded array
		repeated_subarray = subarray.unsqueeze(0).repeat(steps, 1)
		# print("Repeated subarray shape:", repeated_subarray.shape)
		# Both arrays have the same shape now
		# Shape = [#sentences_padded_tokens, #word_tokens]
		# Compare the two arrays:
		comparison = torch.all(unfolded_array == repeated_subarray, dim=-1)
		# print("Comparison:", comparison)
		first_occurrence_index = int(torch.where(comparison == True)[0])
		# We get to a single scalar
		# Now we repeat the first occurrence index (increasing it) for each element of the subarray
		# Shape = [#word_tokens]
		return torch.arange(start=first_occurrence_index, end=first_occurrence_index + window_len, dtype=torch.long)

	def _compute_cross_score(self, sentences_tokens: torch.Tensor, pw_tokens: torch.Tensor | list[int], sw_tokens: torch.Tensor | list[int]) -> float:
		# If the tokens are not tensors, we convert them to tensors
		if isinstance(pw_tokens, list):
			pw_tokens = torch.Tensor(pw_tokens)
		pw_tokens = pw_tokens.long()
		
		if pw_tokens.shape[0] > 1:
			print("WARNING: pw_tokens has more than one token. The method is not tested for this case.")

		# We replace the pw with the mask token
		mask_token_index = []
		for sent_tokens in sentences_tokens:
			pw_indices = self._get_subsequence_index(sent_tokens, pw_tokens)
			sent_tokens[pw_indices] = self.embedder.tokenizer.mask_token_id
			mask_token_index.append(pw_indices)
		mask_token_index = torch.stack(mask_token_index).squeeze()

		with torch.no_grad():
			scores = self.model(input_ids=sentences_tokens, labels=sentences_tokens).logits
			# The resulting scores shape is [#sentences, #tokens, #vocab]
		
		word_scores = scores[:, mask_token_index, :]
		word_scores = word_scores.diagonal(dim1=0, dim2=1).moveaxis(-1, 0)
		# Pairing the two first dimensions, so that we have a tensor of shape (vocabulary_size, batch_size)
		# Then, moving the batch_size dimension to the first position, so that we have a tensor of shape (batch_size, vocabulary_size)

		# Getting the probability of the pw, applying softmax over the vocabulary dimension
		word_probs = self.__softmax(word_scores)
		word_probs = word_probs[:, pw_tokens]

		# Averaging the probabilities over the batch dimension
		avg_probs = word_probs.mean(dim=0)
		return avg_probs