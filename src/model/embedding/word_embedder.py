# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module contains a class used to extract embeddings from a word.
# The class has a core method, called "embed", which takes a word as input (along with some other inputs) and returns the embedding of the word.
# The embedding is a vector of floats, instanced as a PyTorch tensor.
# This process can be done in different ways, depending on what the user wants to do.

import random
import re
import torch
from datasets import Dataset
from datasets.fingerprint import Hasher
from transformers import AutoModel, AutoTokenizer, logging

import sys
from pathlib import Path

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.parent))
from data_processing.sentence_maker import SP_PATTERN, get_dataset_from_words_csv, replace_word
from utils.const import BATCH_SIZE, DEFAULT_BERT_MODEL_NAME, NUM_PROC, DEVICE

# Disabling transformers logging
logging.set_verbosity_error()

class WordEmbedder:
	"""
	This class offers the ability to extract embeddings from a word, through the model BERT.
	In order to extract an embedding, the user must call the "embed" method.
	"""
	def __init__(self, **kwargs) -> None:
		"""
		This method initializes the class.
		Possible keyword arguments are:

			- ``pattern``: the pattern used to find the word in the sentence. 
			By default, it will use the "Stereotype Property" pattern from 'sentence_maker.py'.

			- ``templates_selected_number``: the number of templates to select for each word. 
			If the value is "-1", all the templates will be selected. The selection is done randomly. 
			By default, it will select all the templates.

			- ``average_templates``: whether to average the embeddings of each template or not, for a single word.
			By default, it will average the embeddings (i.e. the value is True).

			- ``average_tokens``: whether to average the embeddings of each token or not, for a single word.
			Note: we don't know in advance if the word will be split into multiple tokens.
			By default, it will average the embeddings (i.e. the value is True).

			- ``max_tokens_number``: the maximum number of tokens to consider for a single word.
			By default, it will consider all the tokens (i.e. the value is "-1").

			- ``discard_longer_words``: whether to discard the words that are longer than the maximum number of tokens.
			By default, it will not discard the words (i.e. the value is False).

		This parameters are used in the "embed" method.

		:param kwargs: The keyword arguments to be passed to the embedding method.
		"""
		# Processing arguments
		# The pattern used to find the word in the sentence.
		if 'pattern' in kwargs:
			self.pattern: re.Pattern[str] = re.compile(kwargs['pattern'])
		else:
			self.pattern: re.Pattern[str] = SP_PATTERN

		# The number of templates to select for each word.
		# If the value is "-1", all the templates will be selected.
		# The selection is done randomly.
		if 'templates_selected_number' in kwargs:
			arg = kwargs['templates_selected_number']
			if arg == 'all':
				self.templates_selected_number: int = -1 # "-1" means all templates will be selected
			else:
				self.templates_selected_number: int = max(1, arg)	# At least one template will be selected
		else:
			self.templates_selected_number: int = -1
		
		# Whether to average the embeddings of each template or not, for a single word
		if 'average_templates' in kwargs:
			self.average_templates = kwargs['average_templates']
		else:
			self.average_templates = True

		# Whether to average the embeddings of each token or not, for a single word
		# Note: we don't know in advance if the word will be split into multiple tokens
		if 'average_tokens' in kwargs:
			self.average_tokens = kwargs['average_tokens']
		else:
			self.average_tokens = True
		
		# The maximum number of tokens to consider for each word
		# If the value is "-1", all the tokens will be considered
		# EXPECTED BEHAVIOUR: When a word is split into multiple tokens:
		#	- if the number of tokens is less or equal than the maximum number of tokens, all the tokens will be considered (and eventually averaged, see "average_tokens")
		#	- if the number of tokens is greater than the maximum number of tokens:
		#		- if the value of "discard_longer_words" is True, the word will be discarded
		#		- if the value of "discard_longer_words" is False, the considered tokens will be the first "max_tokens_number" tokens
		if 'max_tokens_number' in kwargs:
			arg = kwargs['max_tokens_number']
			if arg == 'all':
				self.max_tokens_number: int = -1 # "-1" means all tokens will be considered
			else:
				self.max_tokens_number: int = max(1, arg)	# At least one token has to be considered
		else:
			self.max_tokens_number: int = -1
		
		# Whether to discard the words that are split into more tokens than the maximum number of tokens
		if 'discard_longer_words' in kwargs:
			self.discard_longer_words = kwargs['discard_longer_words']
		else:
			self.discard_longer_words = False

		# The model used to extract the embeddings
		self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BERT_MODEL_NAME)
		self.model = AutoModel.from_pretrained(DEFAULT_BERT_MODEL_NAME).to(DEVICE)

	def get_tokens_number(self, word: str) -> int:
		"""
		This method returns the number of tokens that the word will be split into.

		:param word: The word to be split.
		:return: The number of tokens that the word will be split into.
		"""
		return len(self.tokenizer(word, padding=False, truncation=False, add_special_tokens=False, return_tensors='pt')['input_ids'][0])

	def _get_subsequence_index(self, array: torch.Tensor, subarray: torch.Tensor) -> torch.Tensor:
		# Example:
		# array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		# subarray = [3, 4, 5]
		# output = [2]
		window_len = subarray.shape[-1]
		steps = array.shape[-1] - window_len + 1
		# Unfold the last dimension of the array into 2 dimension of length [len(array) - window_len + 1, window_len]
		unfolded_array = array.unfold(dimension=-1, size=window_len, step=1).to(DEVICE)
		# print("Unfolded array shape:", unfolded_array.shape)
		# Repeat the subarray to match the shape of the unfolded array
		repeated_subarray = subarray.unsqueeze(0).repeat(steps, 1).to(DEVICE)
		# print("Repeated subarray shape:", repeated_subarray.shape)
		# Both arrays have the same shape now
		# Shape = [#sentences_padded_tokens, #word_tokens]
		# Compare the two arrays:
		comparison = torch.all(unfolded_array == repeated_subarray, dim=-1).to(DEVICE)
		# print("Comparison shape:", comparison.shape)
		first_occurrence_index = int(torch.where(comparison == True)[0])
		# We get to a single scalar
		# Now we repeat the first occurrence index (increasing it) for each element of the subarray
		# Shape = [#word_tokens]
		return torch.arange(start=first_occurrence_index, end=first_occurrence_index + window_len, dtype=torch.long).to(DEVICE)

	def _embed_words_batch(self, words: dict[str, list], templates: Dataset) -> list:
		"""
		This method takes a batch of words and computes their embeddings.
		It assumes that the words are already tokenized; the input ``words`` must be a dictionary with the following keys:
		- "word": a list of strings, each string being a word to be embedded.
		- "descriptor": a list of strings, each string being the descriptor of the word. (optional)
		- "value": a list of strings, each string being the value of the word. (optional)
		- "tokens": a list of lists of IDs, each list of strings being the tokens of the word.
		This method is called by the ``embed`` method, which takes care of tokenizing the words.

		The resulting embeddings are a list of PyTorch tensors of size [#templates, #tokens, #features].
		There's a tensor for each word in the batch.
		"""
		templates_list = templates['template']

		def compute_sentences_fn(word_sample):
			""" Getting the sentences for a given word """
			# Creating a list of sentences, where the word was replaced, and filtering out the ones that are not valid
			sentences = map(lambda x: replace_word(sentence=x, word=word_sample, pattern=self.pattern), templates_list)
			sentences = filter(lambda x: x[1] == True, sentences)
			sentences = list(map(lambda x: x[0], sentences))
			# Selecting a random subset of templates/sentences if needed
			if self.templates_selected_number != -1 and self.templates_selected_number < len(sentences):
				random.shuffle(sentences)
				sentences = sentences[:self.templates_selected_number]
			
			if len(sentences) == 0 and word_sample['descriptor'] != 'unused':
				raise Warning(f"Zero (0) sentences were found for the word \"{word_sample['word']}\", which has not an explicit <unused> descriptor.\n" +
				"This may be due to the fact that the word is not present in the templates, or that the word is not present in the templates with the same descriptor.")


			# "sentences" has now the sentences where the word was replaced
			word_sample['sentences'] = sentences
			word_sample['num_sentences'] = len(sentences)
			return word_sample

		# Hashing a function. This is needed for the internal caching of the PyArrow Dataset
		Hasher.hash(compute_sentences_fn)

		# Getting the sentences for each word
		word_with_sentences = Dataset.from_dict(words).map(compute_sentences_fn, batched=False, num_proc=NUM_PROC) #.filter(lambda x: x['num_sentences'] > 0)
		# "word_with_sentences" has now the sentences where the word was replaced, as an item "sentences", and the number of sentences as an item "num_sentences"

		# Flattening the sentences, in order to tokenize them all at once
		all_sentences = [sentence for sentences in word_with_sentences['sentences'] for sentence in sentences]

		# We create a parallel list of tokens, where each element is the list of tokens of the word of the corresponding sentence
		all_word_tokens = [tokens for word in word_with_sentences for tokens in [word['tokens']] * word['num_sentences']]
		assert len(all_sentences) == len(all_word_tokens), "The number of produced sentences for this batch must be the same of the number of tokens groups."

		# Tokenizing the sentences
		tokenized_all_sentences = self.tokenizer(all_sentences, padding=True, truncation=False, return_tensors='pt')['input_ids'].to(DEVICE)
		# "tokenized_all_sentences" is now a tensor of size [#all_sentences, #tokens]
		print("Tokenized all sentences shape: ", tokenized_all_sentences.shape)

		# Getting the indices of the tokens of the words in the sentences
		words_indices = [self._get_subsequence_index(sentence_tokens_ids, torch.Tensor(word_tokens_ids)) for sentence_tokens_ids, word_tokens_ids in zip(tokenized_all_sentences, all_word_tokens)]

		# Getting the embeddings
		embeddings_tensor: torch.Tensor = self.model(tokenized_all_sentences)['last_hidden_state'].to(DEVICE)
		embeddings: list[torch.Tensor] = [embeddings_tensor[sentence_i, word_tokens_indices] for sentence_i, word_tokens_indices in enumerate(words_indices)]

		# Reshaping the embedding list, and grouping them by word
		words_embeddings = []
		for num_sentences_per_word in word_with_sentences['num_sentences']:
			# If the word has no sentences to be embedded, it must be maintained in the same position of the input list, but with an empty tensor
			if num_sentences_per_word == 0:
				words_embeddings.append(torch.empty(0, 1, 768).to(DEVICE))
				# FIXME: the shape of the empty tensor is assuming that the word is tokenized in 1 token only.
				continue

			word_embeddings = torch.stack(embeddings[:num_sentences_per_word]).to(DEVICE)
			# Getting the embeddings of the sentences of this word
			words_embeddings.append(word_embeddings)
			# Removing the embeddings of the sentences of this word
			embeddings = embeddings[num_sentences_per_word:]

		# We're now arrived to a shape [#templates, #tokens, #features] for EACH word
		if self.average_templates:
			words_embeddings = [torch.mean(w_emb, dim=0).unsqueeze(0).to(DEVICE) for w_emb in words_embeddings] 
		if self.average_tokens:
			words_embeddings = [torch.mean(w_emb, dim=1).unsqueeze(1).to(DEVICE) for w_emb in words_embeddings]

		# Based on what average_templates and average_tokens are, the resulting shape is a LIST of tensors of size:
		# - [[#templates, #tokens, #features]]	 if both are False
		# - [[#templates, 1, #features]]		 if average_tokens is True
		# - [[1, #tokens, #features]]			 if average_templates is True
		# - [[1, 1, #features]]					 if both are True
		# However, if the word has no sentences to be embedded, the corresponding tensor has a shape of [0, #tokens, #features]
		return words_embeddings


	def embed(self, words: Dataset, templates: Dataset) -> Dataset:
		"""
		This method returns a dataset where each word is associated with its embedding(s).
		The embedding items are PyTorch tensors of size [#templates, #tokens, #features],
		computed accodingly to the ``embed_word`` method and the initial parameters.

		The returned dataset contains:
		- The original items: "word", "descriptor", and (optional) "value".
		- The following new item: "embedding", which is a PyTorch tensor of size [#templates, #tokens, #features].
		Note that the number of tokens in the embedding can vary, depending on the word and the model vocabulary;
		thus, the tensors in the "embedding" item are not necessarily of the same size.

		IF the ``average_tokens`` parameter is True, the resulting embedding of each word will be of size [#templates, 1, #features].
		Therefore, the 'embedding' column will be memorized as a tensor of size [#words, #templates, 1, #features].
		OTHERWISE, if the ``average_tokens`` parameter is False, the resulting word embedding will be of size [#templates, #tokens, #features].
		Therefore, the 'embedding' column will be memorized as a LIST of tensors [ [#templates, #tokens, #features] ].

		:param words: The dataset of words to be embedded.
		:param templates: The dataset of templates to be used to embed the words.
		:return: The dataset of words with their embeddings.
		"""
		# Checks if the templates are provided as a parameter
		if not templates:
			raise ValueError("The templates must be provided as a parameter.")
		if not words:
			raise ValueError("The words must be provided as a parameter.")

		def tokenize_words_batch_fn(batch: dict[list]) -> dict[list]:
			tokens_ids = self.tokenizer(batch['word'], padding=False, truncation=False, add_special_tokens=False)
			# Note: the special tokens [CLS] and [SEP] are not considered (i.e. they are not added to the tokenized word)
			batch['tokens'] = tokens_ids['input_ids']
			batch['num_tokens'] = [len(tokens) for tokens in tokens_ids['input_ids']]
			# Discarding the words that are too long
			reduced_batch = {}
			tokens_numbers = batch['num_tokens']
			if self.discard_longer_words and self.max_tokens_number != -1:
				for column in batch:
					reduced_batch[column] = [elem for elem, num_tokens in zip(batch[column], tokens_numbers) if num_tokens <= self.max_tokens_number]
				batch = reduced_batch
			return batch

		def embed_words_batch_fn(batch: dict[list]) -> dict[list]:
			# "batch" is a dictionary of lists, where each list contains the values of a column
			# The columns are: ['word', 'value', 'descriptor'(optional), 'tokens']
			embeddings: list[torch.Tensor] = self._embed_words_batch(batch, templates)
			batch['embedding'] = embeddings
			return batch

		# Tokenizing and filtering the words
		words = words.map(tokenize_words_batch_fn, batched=True, batch_size=BATCH_SIZE, num_proc=NUM_PROC)
		# Current columns: ['word', 'value', 'descriptor'(optional), 'tokens', 'num_tokens']

		# Hashing a function. This is needed for the internal caching of the PyArrow Dataset
		Hasher.hash(embed_words_batch_fn)
		# Embedding the words
		embeddings_ds = words.map(embed_words_batch_fn, batched=True, batch_size=BATCH_SIZE, num_proc=NUM_PROC, remove_columns=['num_tokens'])
		# NOTE: the 'num_tokens' column is removed, since it's not needed anymore. If you want to keep it, you can edit the previous line.
		embeddings_ds = embeddings_ds.with_format('torch', device=DEVICE)
		return embeddings_ds


if __name__ == "__main__":
	# Loading the datasets
	templates: Dataset = Dataset.from_csv('data/stereotyped-p/profession/templates-01.csv')
	words: Dataset = get_dataset_from_words_csv('data/stereotyped-p/profession/words-01.csv')

	# Creating the word embedder
	word_embedder = WordEmbedder(select_templates='all', average_templates=False, average_tokens=False, discard_longer_words=True, max_tokens_number=2)

	# Embedding a word
	embedding_dataset = word_embedder.embed(words, templates)
	print(f"Resulting embedding length:", len(embedding_dataset))
