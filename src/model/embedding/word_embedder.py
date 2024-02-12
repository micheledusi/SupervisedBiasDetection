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

import logging
import random
import re
from deprecated import deprecated
import torch
from datasets import Dataset, concatenate_datasets
from datasets.fingerprint import Hasher
from transformers import AutoModel, AutoTokenizer

if __name__ == "__main__":
	import sys
	from pathlib import Path
	directory = Path(__file__)
	sys.path.append(str(directory.parent.parent.parent))
	sys.path.append(str(directory.parent.parent.parent.parent))

from data_processing.sentence_maker import replace_word, replace_word_in_templates
from data_processing.pattern import PATTERN
from utils.config import Configurations, Configurable, Parameter
from utils.const import DEVICE, NUM_PROC, BATCH_SIZE


@deprecated("This class is deprecated. Use the 'RawEmbedder' class instead.")
class WordEmbedder:
	"""
	This class offers the ability to extract embeddings from a word, through the model BERT.
	In order to extract an embedding, the user must call the "embed" method.
	"""
	def __init__(self, configs: Configurations, pattern: re.Pattern[str] | str = PATTERN) -> None:
		"""
		This method initializes the class.

		:param configs: The configurations of the project.
		:param pattern: The pattern used to find the word in the sentence.
		"""
		# Processing arguments
		# The pattern used to find the word in the sentence.
		if isinstance(pattern, str):
			self.pattern: re.Pattern[str] = re.compile(pattern)
		else:
			self.pattern: re.Pattern[str] = pattern

		# The number of templates to select for each word.
		# If the value is "-1", all the templates will be selected.
		# The selection is done randomly.
		arg_tmpl = configs[Parameter.TEMPLATES_PER_WORD_SAMPLING_PERCENTAGE]
		if arg_tmpl == 'all' or arg_tmpl == -1:
			self.templates_selected_number: int = -1 # "-1" means all templates will be selected
		else:
			self.templates_selected_number: int = max(1, arg_tmpl)	# At least one template will be selected
		
		# Whether to average the embeddings of each template or not, for a single word
		self.average_templates = configs.get(Parameter.TEMPLATES_POLICY)

		# Whether to average the embeddings of each token or not, for a single word
		# Note: we don't know in advance if the word will be split into multiple tokens
		self.average_tokens = configs.get(Parameter.AVERAGE_TOKENS)
		
		# The maximum number of tokens to consider for each word
		# If the value is "-1", all the tokens will be considered
		# EXPECTED BEHAVIOUR: When a word is split into multiple tokens:
		#	- if the number of tokens is less or equal than the maximum number of tokens, all the tokens will be considered (and eventually averaged, see "average_tokens")
		#	- if the number of tokens is greater than the maximum number of tokens:
		#		- if the value of "discard_longer_words" is True, the word will be discarded
		#		- if the value of "discard_longer_words" is False, the considered tokens will be the first "max_tokens_number" tokens
		arg_tkns = configs.get(Parameter.MAX_TOKENS_NUMBER)
		if arg_tkns == 'all' or arg_tkns == -1:
			self.max_tokens_number: int = -1
		else:
			self.max_tokens_number: int = max(1, arg_tkns)
		
		# Whether to discard the words that are split into more tokens than the maximum number of tokens
		self.discard_longer_words = configs.get(Parameter.LONGER_WORD_POLICY)

		# The model used to extract the embeddings
		model_name: str = configs[Parameter.MODEL_NAME]
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)	# The tokenizer must be initialized with the "add_prefix_space", specifically for the RoBERTa model. In this case, in fact, the tokenizer will tokenize differently the words at the beginning of each sentence.
		self.model = AutoModel.from_pretrained(model_name).to(DEVICE)

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
		#print("Unfolded array shape:", unfolded_array.shape)
		# Repeat the subarray to match the shape of the unfolded array
		repeated_subarray = subarray.unsqueeze(0).repeat(steps, 1).to(DEVICE)
		#print("Repeated subarray shape:", repeated_subarray.shape)
		# Both arrays have the same shape now
		# Shape = [#sentences_padded_tokens, #word_tokens]
		# Compare the two arrays:
		comparison = torch.all(unfolded_array == repeated_subarray, dim=-1).to(DEVICE)
		#print("Comparison shape:", comparison.shape)
		# Get the first occurrence index
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
			sentences = map(lambda x: replace_word(sentence=x, word_sample=word_sample, pattern=self.pattern), templates_list)
			sentences = filter(lambda x: x[1] == True, sentences)
			sentences = list(map(lambda x: x[0], sentences))
			# Selecting a random subset of templates/sentences if needed
			if self.templates_selected_number != -1 and self.templates_selected_number < len(sentences):
				random.shuffle(sentences)
				sentences = sentences[:self.templates_selected_number]
			
			if len(sentences) == 0:
				raise Warning(f"Zero (0) sentences were found for the word \"{word_sample['word']}\".\n" +
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
		- The original items: "word", "descriptor", and "value".
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

		# Tokenizing and filtering the words
		# Words dataset is: ['word', 'value', 'descriptor'(optional)]

		def tokenize_words_batch_fn(batch: dict[list]) -> dict[list]:
			# We tokenize the single word
			tokens_ids = self.tokenizer(batch['word'], padding=False, truncation=False, add_special_tokens=False)
			# NOTE: the special tokens [CLS] and [SEP] are not considered (i.e. they are not added to the tokenized word)
			batch['tokens'] = tokens_ids['input_ids']
			batch['num_tokens'] = [len(tokens) for tokens in tokens_ids['input_ids']]

			# Checking what to do with the words that are too long
			if self.discard_longer_words and self.max_tokens_number != -1:
				reduced_batch = {}
				tokens_numbers = batch['num_tokens']
				for column in batch:
					reduced_batch[column] = [elem for elem, num_tokens in zip(batch[column], tokens_numbers) if num_tokens <= self.max_tokens_number]
				batch = reduced_batch
			return batch
		
		words = words.map(tokenize_words_batch_fn, batched=True, batch_size=BATCH_SIZE, num_proc=NUM_PROC)
		# Current columns: ['word', 'value', 'descriptor'(optional), 'tokens', 'num_tokens']

		def embed_words_batch_fn(batch: dict[list]) -> dict[list]:
			# "batch" is a dictionary of lists, where each list contains the values of a column
			# The columns are: ['word', 'value', 'descriptor'(optional), 'tokens']
			embeddings: list[torch.Tensor] = self._embed_words_batch(batch, templates)
			batch['embedding'] = embeddings
			return batch

		# Hashing a function. This is needed for the internal caching of the PyArrow Dataset
		Hasher.hash(embed_words_batch_fn)
		# Embedding the words
		embeddings_ds = words.map(embed_words_batch_fn, batched=True, batch_size=BATCH_SIZE, num_proc=NUM_PROC, remove_columns=['num_tokens'])
		# NOTE: the 'num_tokens' column is removed, since it's not needed anymore. If you want to keep it, you can edit the previous line.
		embeddings_ds = embeddings_ds.with_format('torch', device=DEVICE)
		return embeddings_ds
	

class RawEmbedder(Configurable):
	"""
	This class is an alternative to the "WordEmbedder" class.
	Instead of processing the embeddings with combining operations, it simply crosses the words dataset with the templates dataset, 
	and then it tokenizes the sentences and computes the embeddings.

	NOTE: This is an attempt to create a more efficient version of the "WordEmbedder" class.
	It will be merged with the "WordEmbedder" class in the future, if it will be successful.
	"""
	def __init__(self, configs: Configurations, pattern: re.Pattern[str] | str = PATTERN) -> None:
		# Calling the superclass constructor
		# The constructor of superclass `Configured` declares the list of parameters that are used in the class.
		# This is an additional check to ensure that no parameter outside of the list is used.
		super(RawEmbedder, self).__init__(configs, parameters=[
			Parameter.MODEL_NAME,
			Parameter.MAX_TOKENS_NUMBER,
			Parameter.LONGER_WORD_POLICY,
			])
		
		# Saving the pattern used to find the word in the sentence
		if isinstance(pattern, str):
			self.pattern: re.Pattern[str] = re.compile(pattern)
		else:
			self.pattern: re.Pattern[str] = pattern

		# The model used to extract the embeddings
		self.tokenizer = AutoTokenizer.from_pretrained(self.configs[Parameter.MODEL_NAME], add_prefix_space=True)	# The tokenizer must be initialized with the "add_prefix_space", specifically for the RoBERTa model. In this case, in fact, the tokenizer will tokenize differently the words at the beginning of each sentence.
		self.model = AutoModel.from_pretrained(self.configs[Parameter.MODEL_NAME]).to(DEVICE)
	

	def embed(self, words: Dataset, templates: Dataset) -> Dataset:
		"""
		This method returns a dataset where each row is a combination of a word and a dataset. 
		This combination contains the embedding(s) associated to it.

		The value for the "embedding" column in each row is a PyTorch tensor of size [#features],
		which refers to the embeddings of the word within the associated template. If the word is split into multiple tokens, the tokens are averaged.
		For this reason, the 'tokens' dimension does not appear in the returned dataset.

		In details, the returned dataset contains:
		- The original items: "word", "descriptor", "value" (optional), and "template".
		- The following new item: "embedding", which is a PyTorch tensor of size [#features]. Each row has always a single tensor of the same size.

		:param words: The dataset of words to be embedded.
		:param templates: The dataset of templates to be used to embed the words.
		:return: The combinations (words x templates) with their embeddings.
		"""
		# Checks if the templates are provided as a parameter
		if not templates:
			raise ValueError("The templates must be provided as a parameter.")
		if not words:
			raise ValueError("The words must be provided as a parameter.")

		# XXX: STEP 1 #
		# Tokenizing and filtering the words
		# Words dataset is: ['word', 'value', 'descriptor'(optional)]

		def tokenize_batch_of_words_fn(batch: dict[str, list]) -> dict[str, list]:
			# We tokenize the single words, not considering the templates
			tokens_ids = self.tokenizer(batch['word'], padding=False, truncation=False, add_special_tokens=False)
			# Note: the special tokens [CLS] and [SEP] are not considered (i.e. they are not added to the tokenized word)
			batch['tokens'] = tokens_ids['input_ids']
			batch['num_tokens'] = [len(tokens) for tokens in tokens_ids['input_ids']]
			
			# Checking what to do with the words that are too long
			# If the policy for longer words is "discard"
			if self.configs[Parameter.LONGER_WORD_POLICY] == 'discard':
				# If the maximum number of accepted tokens is not "all" and not "-1"
				max_tokens_number = self.configs[Parameter.MAX_TOKENS_NUMBER]
				if max_tokens_number != 'all' \
						and max_tokens_number != -1:
					# We discard the words that are too long
					reduced_batch = {}
					tokens_numbers = batch['num_tokens']
					for column in batch:
						reduced_batch[column] = [elem for elem, num_tokens in zip(batch[column], tokens_numbers) if num_tokens <= max_tokens_number]
					batch = reduced_batch

			# If the policy for longer words is "truncate"
			elif self.configs[Parameter.LONGER_WORD_POLICY] == 'truncate':
				# If the maximum number of accepted tokens is not "all" and not "-1"
				max_tokens_number = self.configs[Parameter.MAX_TOKENS_NUMBER]
				if max_tokens_number != 'all' \
						and max_tokens_number != -1:
					# We truncate the words that are too long
					batch['tokens'] = [tokens[:max_tokens_number] for tokens in batch['tokens']]
					batch['num_tokens'] = [min(num_tokens, max_tokens_number) for num_tokens in batch['num_tokens']]

					# TODO: Verify whether this is used in the templated embeddings

			# If the policy for longer words is "ignore"
			elif self.configs[Parameter.LONGER_WORD_POLICY] == 'ignore':
				pass	

			# If the policy for longer words is neither "discard" nor "truncate" nor "ignore"
			else:
				raise ValueError(f"The policy for longer words must be either 'truncate' or 'discard', not '{self.configs[Parameter.LONGER_WORD_POLICY]}'.")
			
			return batch
		
		# Applying the tokenization and filtering
		logging.info(f"Tokenizing and filtering the words...")
		words = words.map(tokenize_batch_of_words_fn, batched=True, batch_size=BATCH_SIZE, num_proc=NUM_PROC)
		logging.info(f"Tokenizing and filtering the words... DONE")
		logging.info(f"Resulting words dataset size: {len(words)}")
		# Current columns in words dataset: ['word', 'value', 'descriptor'(optional), 'tokens', 'num_tokens']

		# XXX: STEP 2 #
		# We create the sentences for each word, by replacing the word in the templates
		# Current columns in templates dataset: ['template']

		def cross_batch_of_words_with_templates_fn(batch: dict[str, list]) -> dict[str, list]:
			""" This function creates the sentences for each word in the batch.
			It takes as input the batch of words. For each word in the batch, it creates the sentences where the word was replaced.
			Each row of the batch, so, turns into a series of rows where the word was replaced in the templates.
			NOTE: only the valid sentences are kept. If a word and a template are not compatible, the sentence is discarded.

			The input batch is a dictionary with the following keys:
			- "word": a list of strings, each string being a word to be embedded.
			- "descriptor": a list of strings, each string being the descriptor of the word. (optional)
			- "value": a list of strings, each string being the value of the word. (optional)
			- "tokens": a list of lists of IDs, each list of strings being the tokens of the word.
			- "num_tokens": a list of integers, each integer being the number of tokens of the word.

			The output batch is a dictionary with the following keys:
			- "word": a list of strings, each string being a word to be embedded.
			- "descriptor": a list of strings, each string being the descriptor of the word. (optional)
			- "value": a list of strings, each string being the value of the word. (optional)
			- "tokens": a list of lists of IDs, each list of strings being the tokens of the word.
			- "num_tokens": a list of integers, each integer being the number of tokens of the word.
			- "template": a list of strings, each string being a template.
			- "sentence": a list of strings, each string being a sentence where the word was replaced.
			"""
			# Creating a list of sentences, where the word was replaced, and filtering out the ones that are not valid
			words_batch_ds: Dataset = Dataset.from_dict(batch)
			sentences_batch_ds_list: list[Dataset] = []

			for word in words_batch_ds:
				# For a single word, we create the sentences where the word was replaced
				# FIXME: This is a slow implementation, but it's maybe the most straightforward one
				word_sentences: Dataset = replace_word_in_templates(word, templates, self.pattern)

				# Checking if the word has been replaced in at least one sentence
				if len(word_sentences) == 0:
					# If not, the sentences are unusable. We raise a warning.
					raise Warning(f"Zero (0) sentences were found for the word \"{word['word']}\".\n" +
					"This may be due to the fact that the word is not present in the templates, or that the word is not present in the templates with the same descriptor.")
				
				# Cloning the word sample for each sentence, and adding the word columns to the sentences
				for col in word:
					word_sentences = word_sentences.add_column(col, [word[col]] * len(word_sentences))

				# Insert the sentences for this word in the list
				sentences_batch_ds_list.append(word_sentences)
			
			# Concatenating the sentences for each word
			ds: Dataset = concatenate_datasets(sentences_batch_ds_list)
			return ds.to_dict()

		# Getting the sentences for each word
		logging.info(f"Creating the sentences for each word...")
		word_with_sentences: Dataset = words.map(cross_batch_of_words_with_templates_fn, batched=True, batch_size=BATCH_SIZE, num_proc=NUM_PROC)\
			.with_format('torch', device=DEVICE)
		logging.info(f"Creating the sentences for each word... DONE")
		logging.info(f"Resulting word_with_sentences dataset size: {len(word_with_sentences)}")
		# Current columns in word_with_sentences dataset: ['word', 'value', 'descriptor', 'tokens', 'num_tokens', 'template', 'sentence']

		# XXX: STEP 3 #
		# Tokenizing the sentences and computing the embeddings

		def embed_sentences_batch_fn(sentences_batch: dict[str, list]) -> dict[str, list]:
			"""
			This method takes a batch of sentences and computes their embeddings.
			The input ``sentences_batch`` must be a dictionary of lists, with the following keys:
			- "word": the words to be embedded.
			- "descriptor": the descriptors of the word.
			- "value": the values of the word.
			- "template": the templates to be used to embed the words.
			- "sentence": the strings obtained by instantiating the words within the templates.
			- "tokens": the IDs of the tokens of each word.
			- "num_tokens": the number of tokens of each word.

			The result is the same batch, with the addition of the "embedding" key, which contains the embeddings of the words.
			"""
			# Tokenizing the sentences
			tokenized_sentences: torch.Tensor = self.tokenizer(sentences_batch['sentence'], padding=True, truncation=False, return_tensors='pt')['input_ids'].to(DEVICE)
			batch_len: int = tokenized_sentences.shape[0]
			# We obtain a tensor of size [#sentences, #tokens] containing the IDs of the tokens of each sentence

			# Now we extract only the tokens IDs of the embedded words
			words_indices: list = [self._get_subsequence_index(sentence_tokens, word_tokens) 
							for sentence_tokens, word_tokens 
							in zip(tokenized_sentences, sentences_batch['tokens'])]
			assert len(words_indices) == batch_len, "The number of words_indices must be equal to the batch size."
			
			# Computing the embeddings
			embeddings_tensor: torch.Tensor = self.model(tokenized_sentences)['last_hidden_state'].to(DEVICE)
			assert embeddings_tensor.shape[0] == batch_len, "The number of embedded sentences must be equal to the batch size."
			
			# Extracting the embeddings of the word from the tensor
			embeddings: list[torch.Tensor] = [embeddings_tensor[sentence_i, word_j] for sentence_i, word_j in enumerate(words_indices)]
			# Averaging the embeddings of different tokens of the same word in the same sentence
			embeddings = [torch.mean(emb, dim=0) for emb in embeddings]

			# Adding the embeddings to the batch
			sentences_batch['embedding'] = embeddings
			return sentences_batch

		# Embedding the words by applying the "embed_sentences_batch_fn" function
		logging.info(f"Embedding the words...")
		embeddings_ds = word_with_sentences\
			.map(embed_sentences_batch_fn, batched=True, batch_size=BATCH_SIZE, num_proc=NUM_PROC)\
				.remove_columns(['tokens', 'num_tokens'])\
					.with_format('torch', device=DEVICE)
		logging.info(f"Embedding the words... DONE")
		logging.info(f"Resulting embedding dataset size: {len(embeddings_ds)}")

		# Current columns in embeddings_ds dataset: ['word', 'value', 'descriptor', 'template', 'sentence', 'embedding']
		return embeddings_ds


	def _get_subsequence_index(self, array: torch.Tensor, subarray: torch.Tensor) -> torch.Tensor:
		# Example:
		# array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		# subarray = [3, 4, 5]
		# output = [2]
		window_len = subarray.shape[-1]
		steps = array.shape[-1] - window_len + 1
		# Unfold the last dimension of the array into 2 dimension of length [len(array) - window_len + 1, window_len]
		unfolded_array = array.unfold(dimension=-1, size=window_len, step=1).to(DEVICE)
		#print("Unfolded array shape:", unfolded_array.shape)
		# Repeat the subarray to match the shape of the unfolded array
		repeated_subarray = subarray.unsqueeze(0).repeat(steps, 1).to(DEVICE)
		#print("Repeated subarray shape:", repeated_subarray.shape)
		# Both arrays have the same shape now
		# Shape = [#sentences_padded_tokens, #word_tokens]
		# Compare the two arrays:
		comparison = torch.all(unfolded_array == repeated_subarray, dim=-1).to(DEVICE)
		#print("Comparison shape:", comparison.shape)
		# Get the first occurrence index
		first_occurrence_index = int(torch.where(comparison == True)[0])
		# We get to a single scalar
		# Now we repeat the first occurrence index (increasing it) for each element of the subarray
		# Shape = [#word_tokens]
		return torch.arange(start=first_occurrence_index, end=first_occurrence_index + window_len, dtype=torch.long).to(DEVICE)


if __name__ == "__main__":

	configs = Configurations({
		Parameter.MODEL_NAME: "bert-base-uncased",
		Parameter.MAX_TOKENS_NUMBER: 'all',
		Parameter.LONGER_WORD_POLICY: 'ignore',
	})
		
	# Loading the datasets
	templates: Dataset = Dataset.from_csv('data/properties/religion/templates-01.csv')
	words: Dataset = Dataset.from_csv('data/properties/religion/words-01.csv')

	word_embedder = RawEmbedder(configs, PATTERN)
	embedding_dataset = word_embedder.embed(words, templates)

	print("Resulting embedding dataset:", embedding_dataset)
	print("Resulting embedding length:", len(embedding_dataset))