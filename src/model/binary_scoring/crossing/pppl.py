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


IGNORE_INDEX: int = -100


class PPPLCrossScorer(CrossingScorer):
	"""
	This class implements the CrossScorer interface for Masked Language Models.
	It measures the Pseudo-Perplexity (PPPL) score for sentences where a pair of word (protected and stereotyped) is involved.

	For example, given the sentence "She is a nurse.", the PPPL score is computed as follows:
	1. The sentence is tokenized: [CLS] [She] [is] [a] [nurse] [SEP].
	2. Each non-special token is masked, and the model is asked to predict each masked token:
		- [CLS] [_MASK_] [is] [a] [nurse] [SEP]
		- [CLS] [She] [_MASK_] [a] [nurse] [SEP]
		- [CLS] [She] [is] [_MASK_] [nurse] [SEP]
		- [CLS] [She] [is] [a] [_MASK_] [SEP]
	3. Each prediction gives a probability, from which we have the probability of the correct token.
	4. We take the negative log of each probability, and we average them. The exponential of the average is the PPPL score for the sentence.

	In the end, the output score is the average of the PPPL scores of all the sentences given to the model.

	"""

	def __init__(self, configs: Configurations):
		super().__init__(configs)
		# Getting the model
		model_name: str = configs[Parameter.MODEL_NAME]
		self.model = AutoModelForMaskedLM.from_pretrained(model_name)
		self.model.to(DEVICE)
	
	def _compute_pppl(self, sentence: torch.Tensor) -> float:
		"""
		This method computes the Pseudo-Perplexity (PPPL) of a sentence,
		by taking the exponential of the average of the negative log-probabilities of each token in the sentence.

		A high PPPL score means that the sentence is LESS likely to be generated by the model, 
		which means that the sentence makes the model more "perplexed".

		:param sentence: the sentence to be scored.
		:return: the PPPL of the sentence.
		"""
		tensor_input = sentence[sentence.nonzero()].squeeze()
		# tensor([[ 101, A,	B, C, 102]])
		repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1).to(DEVICE)
		# tensor([[ 101, A, B, C, 102],
		#         [ 101, A, B, C, 102],
		#         [ 101, A, B, C, 102]])
		mask = torch.ones(tensor_input.size(-1) - 1, device=DEVICE).diag(1)[:-2]
		# tensor([[0., 1., 0., 0., 0.],
		#         [0., 0., 1., 0., 0.],
		#         [0., 0., 0., 1., 0.]])
		masked_input = repeat_input.masked_fill(mask == 1, self.embedder.tokenizer.mask_token_id).to(DEVICE)
		# tensor([[ 101, 103,   B,   C, 102],
		#         [ 101,   A, 103,   C, 102],
		#         [ 101,   A,   B, 103, 102]])
		labels = repeat_input.masked_fill(masked_input != self.embedder.tokenizer.mask_token_id, IGNORE_INDEX).to(DEVICE)
		# tensor([[-100,    A, -100, -100, -100],
		#         [-100, -100,    B, -100, -100],
		#         [-100, -100, -100,    C, -100]])
		# The "-100" is a special value for the loss function, which ignores it.
		results = self.model(masked_input, labels=labels)
		loss = results['loss']
		# Taking the loss from the model output is equivalent to compute the PPPL "by hand".
		# The loss function is the CrossEntropyLoss, which is the negative log-likelihood.
		# The following code is equivalent to the previous one.
		#	import torch.nn.functional as F
		# 	logits = results['logits']
		#	vocab_size = logits.size(-1)
		#	loss2 = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=IGNORE_INDEX, reduction='mean')
		score: float = torch.exp(loss).item()
		return score

	def _compute_crossing_score(self, sentences_tokens: torch.Tensor, pw_tokens: torch.Tensor | list[int], sw_tokens: torch.Tensor | list[int]) -> float:
		# The score is the average of the PPPL of each sentence.
		scores_sum: float = 0
		for sentence in sentences_tokens:
			scores_sum += self._compute_pppl(sentence)
		return scores_sum / len(sentences_tokens)