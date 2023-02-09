# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module contains the methods used to generate the "cross-scores".
# The cross-scores are scores computed on pairs of words. 
# In particular, each cross-score involves a word from the stereotyped property 
# and a word from the protected property.

from abc import abstractmethod
from datasets import Dataset
import torch
from tqdm import tqdm

from data_processing.sentence_maker import replace_protected_word, replace_stereotyped_word
from model.embedding.word_embedder import WordEmbedder
from utils.const import DEFAULT_DISCARD_LONGER_WORDS, DEFAULT_MAX_TOKENS_NUMBER


class CrossScorer:

	def __init__(self, **kwargs) -> None:
		"""
		Initializes the CrossScorer object.
		In the initialization, the following parameters can be specified:
		- max_tokens_number: the maximum number of tokens to consider/retain for the word.
		- discard_longer_words: whether to discard the words that are split into more tokens than the maximum number of tokens.

		If the parameters are not specified, the default values are used (in "const.py").
		If the "max_tokens_number" is set to "all", all the tokens will be considered and the "discard_longer_words" parameter will be ignored.
		Otherwise, if the "max_tokens_number" is set to a value greater than 1, a word with more tokens than the maximum number of tokens:
		- will be *discarded* if "discard_longer_words" is set to True;
		- will be *truncated* if "discard_longer_words" is set to False.

		:param kwargs: The arguments for the initialization.
		"""
		# The maximum number of tokens to consider/retain for the word
		if 'max_tokens_number' in kwargs:
			arg = kwargs['max_tokens_number']
			if arg == 'all':
				self.max_tokens_number: int = -1 # "-1" means all tokens will be considered
			else:
				self.max_tokens_number: int = max(1, arg)	# At least one token has to be considered
		else:
			self.max_tokens_number: int = DEFAULT_MAX_TOKENS_NUMBER
		
		# Whether to discard the words that are split into more tokens than the maximum number of tokens
		if 'discard_longer_words' in kwargs:
			self.discard_longer_words = kwargs['discard_longer_words']
		else:
			self.discard_longer_words = DEFAULT_DISCARD_LONGER_WORDS
		
		# Initializing the embedder and the tokenizer
		self.embedder = WordEmbedder()

	def compute_cross_scores(self, templates: Dataset, protected_words: Dataset, stereotyped_words: Dataset) -> tuple[Dataset, Dataset, torch.Tensor]:
		"""
		This method computes the cross-scores for each pair of words.
		The cross-scores are computed for each pair of words in the protected_words and stereotyped_words datasets.

		:param templates: The templates dataset.
		:param protected_words: The protected words dataset.
		:param stereotyped_words: The stereotyped words dataset.
		:return: a tuple containing:
		- The updated protected_words dataset, with the "tokens" column added, and filtered by max_tokens_number and discard_longer_words.
		- The updated stereotyped_words dataset, with the "tokens" column added, and filtered by max_tokens_number and discard_longer_words.
		- The cross-scores tensor, with size (#protected_words, #stereotyped_words).
		"""
		# Preparing an auxiliary embedder that can be used to discard long words (i.e. words that are tokenized in more than X tokens)
		if self.discard_longer_words and isinstance(self.max_tokens_number, int) and self.max_tokens_number > 0:
			print("Filtering the stereotyped words...")
			stereotyped_words = stereotyped_words.filter(lambda x: self.embedder.get_tokens_number(x['word']) <= self.max_tokens_number)

		print("Preparing word tokens...")
		max_lenght = self.max_tokens_number if isinstance(self.max_tokens_number, int) and self.max_tokens_number > 0 else None
		# Preparing the tokens for the protected words
		protected_tokens: list[list[int]] = self.embedder.tokenizer(protected_words['word'], padding=False, truncation=True, max_length=max_lenght, add_special_tokens=False)['input_ids']
		protected_words = protected_words.add_column('tokens', protected_tokens)
		# Preparing the tokens for the stereotyped words
		stereotyped_tokens: list[list[int]] = self.embedder.tokenizer(stereotyped_words['word'], padding=False, truncation=True, max_length=max_lenght, add_special_tokens=False)['input_ids']
		stereotyped_words = stereotyped_words.add_column('tokens', stereotyped_tokens)
		
		print("Computing the scores...")		
		words_scores: torch.Tensor = torch.zeros(size=(len(protected_words), len(stereotyped_words)))

		# Computing the scores for each pair of words
		for i, sw in tqdm(enumerate(stereotyped_words), total=len(stereotyped_words)):
			# For each template, we insert the stereotyped word in the slot
			# And we filter only the sentences where the replacement has been performed
			sw_sentences_pairs = [replace_stereotyped_word(sent, sw) for sent in templates['template']]
			sw_sentences = [sent_pair[0] for sent_pair in sw_sentences_pairs if sent_pair[1] is True]
			
			for j, pw in enumerate(protected_words):
				# For each protected word, we try to replace it in the sentence
				# And we filter only the sentences where the replacement has been performed
				pw_sentences_pairs = [replace_protected_word(sent, pw) for sent in sw_sentences]
				pw_sentences = [sent_pair[0] for sent_pair in pw_sentences_pairs if sent_pair[1] is True]

				if len(pw_sentences) == 0:
					# If no sentence has been found, we skip the pair
					continue

				# We now have a series of sentences where both words (protected and stereotyped) have been replaced
				# We tokenize the sentences
				sentences_tokens = self.embedder.tokenizer(pw_sentences, padding=True, truncation=False, return_tensors='pt')['input_ids']
				# Retrieving the tokens for the protected and stereotyped words
				pw_tokens = pw['tokens']
				sw_tokens = sw['tokens']
				
				# Computing the cross-score
				score = self._compute_cross_score(sentences_tokens, pw_tokens, sw_tokens)
				words_scores[j, i] = score
		
		return protected_words, stereotyped_words, words_scores

	def average_by_values(self, protected_words: Dataset, stereotyped_words: Dataset, words_scores: torch.Tensor) -> tuple[tuple, tuple, torch.Tensor]:
		"""
		This method computes the average of the cross-scores for each pair of words, grouped by:
		- the value of the stereotyped word, and
		- the value of the protected word.

		:param protected_words: The protected words dataset.
		:param stereotyped_words: The stereotyped words dataset.
		:param words_scores: The cross-scores tensor, with size (#protected_words, #stereotyped_words).
		:return: A tuple containing:
		- A tuple containing the protected values.
		- A tuple containing the stereotyped values.
		- The average scores tensor, with size (#protected_values, #stereotyped_values).
		"""
		# Preparing the tensor with size (#protected_values, #stereotyped_values)
		protected_values = tuple(set(protected_words['value']))
		stereotyped_values = tuple(set(stereotyped_words['value']))
		average_scores = torch.zeros(size=(len(protected_values), len(stereotyped_values)))

		for pv_index, pv in enumerate(protected_values):
			for sv_index, sv in enumerate(stereotyped_values):
				# Computing the average of the cross-scores for each pair of words, grouped by:
				# - the value of the stereotyped word, and
				# - the value of the protected word.
				pw_indices = [i for i, x in enumerate(protected_words['value']) if x == pv]
				sw_indices = [i for i, x in enumerate(stereotyped_words['value']) if x == sv]
				average_scores[pv_index, sv_index] = words_scores[pw_indices][:, sw_indices].mean()
		
		return protected_values, stereotyped_values, average_scores
	
	@abstractmethod
	def _compute_cross_score(self, sentences_tokens: torch.Tensor, pw_tokens: torch.Tensor | list[int], sw_tokens: torch.Tensor | list[int]) -> float:
		"""
		This method computes the cross-score for a pair of words, in a series of sentences.

		:param sentences_tokens: The tokens of the sentences, with size (#sentences, #tokens).
		:param pw_tokens: The tokens of the protected word.
		:param sw_tokens: The tokens of the stereotyped word.
		:return: The cross-score, as a float.
		"""
		raise NotImplementedError("This method must be implemented by the subclasses.")