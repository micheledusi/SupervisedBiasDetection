# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

import torch
from transformers import AutoModelForMaskedLM

from model.binary_scoring.crossing.base import CrossingScorer
from utils.config import Configurations, Parameter
from utils.const import DEVICE

VERBOSE: bool = False


class MLMCrossScorer(CrossingScorer):
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

	def __init__(self, configs: Configurations):
		super().__init__(configs)
		# Getting the model
		model_name: str = configs[Parameter.MODEL_NAME]
		self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(DEVICE)

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

	def _compute_crossing_score(self, sentences_tokens: torch.Tensor, pw_tokens: torch.Tensor | list[int], sw_tokens: torch.Tensor | list[int]) -> float:
		# If the tokens are not tensors, we convert them to tensors
		if isinstance(pw_tokens, list):
			pw_tokens = torch.Tensor(pw_tokens)
		pw_tokens = pw_tokens.long()

		if VERBOSE:
			print("\n\n\nSentences tokens shape:", sentences_tokens.shape)
			print("PW tokens shape:", pw_tokens.shape)

		# We replace the pw with the mask token
		mask_token_index = []
		for sent_tokens in sentences_tokens:
			pw_indices = self._get_subsequence_index(sent_tokens, pw_tokens)
			sent_tokens[pw_indices] = self.embedder.tokenizer.mask_token_id
			mask_token_index.append(pw_indices)
			if VERBOSE:
				print("Sentence: ", sent_tokens)
				print("Protected word tokens:", pw_tokens)
				print("Protected word indices:", pw_indices)
				print("Sentence with mask:", sent_tokens)
		
		mask_token_index = torch.stack(mask_token_index)
		if VERBOSE:
			print("Mask token index shape:", mask_token_index.shape)
			print("Mask token index:", mask_token_index)
			print("Resulting sentences:", sentences_tokens)
		# Shape = [#sentences, #pw_tokens]
		
		
		# Computing the scores for the sentences
		with torch.no_grad():
			scores: torch.Tensor = self.model(input_ids=sentences_tokens, labels=sentences_tokens).logits
			if VERBOSE:
				print("Scores shape:", scores.shape)
			# The resulting scores shape is [#sentences, #all_tokens, #vocab]

			# We define the new shape as [#sentences, #pw_tokens, #vocab]
			word_scores_shape = (scores.shape[0], pw_tokens.shape[0], scores.shape[-1])
			# Then, we get the scores for the pw tokens for each sentence
			word_scores = torch.zeros(word_scores_shape, dtype=torch.float32)
			for sent_index in range(scores.shape[0]):
				word_scores[sent_index] = scores[sent_index, mask_token_index[sent_index]]
			
			if VERBOSE:
				print("Word scores shape:", word_scores.shape)
			# The shape is now [#sentences, #pw_tokens, #vocab]

			# Now we compute the probabilities for each tokens of the pw for each sentence
			word_probs = self.__softmax(word_scores)
			if VERBOSE:
				print("Word probs shape:", word_probs.shape)
			# The shape is still [#sentences, #pw_tokens, #vocab]

			# On the vocabulary dimension, we keep only the probabilities for the pw tokens
			word_probs = word_probs[:, :, pw_tokens]
			if VERBOSE:
				print("Word probs shape:", word_probs.shape)
			# The shape is now [#sentences, #pw_tokens, #pw_tokens]
			# The last two dimensions mean: what is the probability of the X token to be in the spot of the Y token?
			# For this, we keep only the diagonal of the matrix, where X = Y
			word_probs = word_probs.diagonal(dim1=-2, dim2=-1)
			if VERBOSE:
				print("Word probs shape:", word_probs.shape)
			# The shape is now [#sentences, #pw_tokens]

			# Now we can choose to manage the resulting values in different ways:
			# - We can average the probabilities over the batch dimension	
			# - We can average the probabilities over the pw tokens dimension
			# - We can do both.
			# We choose to consider the final MLM score to be the average of the probabilities for EACH token of the protected word, for EACH sentence.
			avg_probs = word_probs.mean()
			if VERBOSE:
				print("Avg probs:", avg_probs)
			return avg_probs

