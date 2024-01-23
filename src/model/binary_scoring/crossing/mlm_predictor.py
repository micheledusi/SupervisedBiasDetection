# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	2023				#
# - - - - - - - - - - - - - - - #

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

import sys
from pathlib import Path
directory = Path(__file__)
sys.path.append(str(directory.parent.parent.parent))
from utils.const import *
from data_processing.sentence_maker import replace_stereotyped_word, mask_protected_word
from data_processing.sentence_maker import get_generation_datasets
from model.embedding.word_embedder import WordEmbedder
from utils.config import Configurations, Parameter


class MLMPredictor:
	"""
	This class contains the methods for predicting a protected property value using a masked language model.
	"""

	__softmax = torch.nn.Softmax(dim=-1)

	def __init__(self, configs: Configurations) -> None:
		model_name: str = configs[Parameter.MODEL_NAME]
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(DEVICE)
		
		# The maximum number of tokens to consider/retain for the word
		arg_tkns = configs.get(Parameter.MAX_TOKENS_NUMBER, DEFAULT_MAX_TOKENS_NUMBER)
		if arg_tkns == 'all' or arg_tkns == -1:
			self.max_tokens_number: int = -1
		else:
			self.max_tokens_number: int = max(1, arg_tkns)
		
		# Whether to discard the words that are split into more tokens than the maximum number of tokens
		self.discard_longer_words: bool = configs.get(Parameter.DISCARD_LONGER_WORDS, DEFAULT_DISCARD_LONGER_WORDS)

	def predict(self, sentences: str | list[str], target: str) -> torch.Tensor:
		"""
		This method predicts the probability that the target word is inserted in the token MASK slot.
		If multiple sentences are provided, the method will return a tensor with the probabilities for each sentence.

		:param sentences: The sentence or the sentences list containing the masked word.
		:param target: The target word to be masked.
		:return: The predicted value of the protected property.
		"""
		# Asserting that the sentence is a list of strings
		# Each sentence will be a batch in the model
		if isinstance(sentences, str):
			sentences = [sentences]

		# Asserting that each sentence contains the MASK token
		for sentence in sentences:
			if TOKEN_MASK not in sentence:
				raise Exception(f"The sentence '{sentence}' does not contain the mask token ({TOKEN_MASK}).")
			
		inputs = self.tokenizer(sentences, padding=True, truncation=False, return_tensors="pt")
		mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]	# Getting the index of the mask token for each sentence (=batch)

		# Processing the inputs with the model
		with torch.no_grad():
			token_logits = self.model(**inputs).logits
			# The shape of the tensor is (batch_size, sequence_length (padded), vocabulary_size)

		mask_token_logits = token_logits[:, mask_token_index, :]
		# This gives us the score for every token in the vocabulary, as if it were the target word
		# The shape of the tensor is (batch_size, batch_size, vocabulary_size)

		mask_token_logits = mask_token_logits.diagonal(dim1=0, dim2=1).moveaxis(-1, 0)
		# Pairing the two first dimensions, so that we have a tensor of shape (vocabulary_size, batch_size)
		# Then, moving the batch_size dimension to the first position, so that we have a tensor of shape (batch_size, vocabulary_size)
		
		mask_token_scores = self.__softmax(mask_token_logits)
		# This gives us the probability for every token in the vocabulary, applying the softmax function on the last dimension
		# The shape of the tensor is (batch_size, vocabulary_size)

		# Getting and returning the scores for the target word (one for each sentence)
		target_index = self.tokenizer.convert_tokens_to_ids(target)
		word_scores = mask_token_scores[:, target_index]
		return word_scores

	def compute_scores(self, protected_property: str, stereotyped_property: str, generation_file_id: int = 1) -> Dataset:
		# Retrieve the datasets for MLM
		pp_words, sp_words, templates = get_generation_datasets(protected_property, stereotyped_property, generation_file_id)

		# Preparing an auxiliary embedder that can be used to discard long words (i.e. words that are tokenized in more than X tokens)
		embedder = WordEmbedder()
		if self.discard_longer_words and isinstance(self.max_tokens_number, int) and self.max_tokens_number > 0:
			print("Filtering the stereotyped words...")
			sp_words = sp_words.filter(lambda x: embedder.get_tokens_number(x['word']) <= self.max_tokens_number)
		
		print("Computing the scores...")
		resulting_scores: dict[str, list] = {'stereotyped_word': [], 'stereotyped_value': []}
		# Creating an empty scores list for each protected value
		for pw in pp_words:
			resulting_scores[pw['value']] = []

		# For each stereotyped word
		for sw in tqdm(sp_words):
			# For each template, we insert the stereotyped word in the slot
			sentences_pairs = [replace_stereotyped_word(sent, sw) for sent in templates['template']]
			sentences = [sent_pair[0] for sent_pair in sentences_pairs if sent_pair[1] is True]

			# Prepare the dictionary for storing the scores for each protected value
			resulting_scores["stereotyped_word"].append(sw['word'])
			resulting_scores["stereotyped_value"].append(sw['value'])

			# For each protected value, we store the scores
			partial_scores: dict[str, torch.Tensor] = {}
			for pw in pp_words:
				# For each protected word, we try to replace it in the sentence
				masked_sentences_pairs = [mask_protected_word(sent, pw) for sent in sentences]
				masked_sentences = [sent_pair[0] for sent_pair in masked_sentences_pairs if sent_pair[1] is True]

				# Then, we predict the probability that the protected word is inserted in the MASK slot
				# We obtain a score for each sentence
				scores: torch.Tensor = self.predict(masked_sentences, pw['word'])

				# Extending the scores list for the current protected value
				protected_value = pw['value']
				if protected_value not in partial_scores:
					partial_scores[protected_value] = scores
				else:
					partial_scores[protected_value] = torch.cat((partial_scores[protected_value], scores))

			# For each protected value
			for protected_value in partial_scores:
				# We compute the mean score for the current stereotyped value
				mean_score = torch.mean(partial_scores[protected_value]).item()
				# And we append it to the list of scores
				resulting_scores[protected_value].append(mean_score)

		# Re-formatting
		resulting_scores: Dataset = Dataset.from_dict(resulting_scores).with_format("torch", device=DEVICE)

		# Pairing the scores relative to two protected values, and computing the polarization
		values_columns = resulting_scores.column_names[2:]
		for i, column1 in enumerate(values_columns):
			for column2 in values_columns[i+1:]:
				if column1 != column2:
					values: torch.Tensor = resulting_scores[column1].sub(resulting_scores[column2])
					resulting_scores = resulting_scores.add_column(f"polarization_{column1}_{column2}", values.tolist())
		
		return resulting_scores


if __name__ == "__main__":
	# Constants
	PROTECTED_PROPERTY: str = "religion"
	STEREOTYPED_PROPERTY: str = "quality"

	# Computing the scores
	resulting_scores = MLMPredictor().compute_scores(PROTECTED_PROPERTY, STEREOTYPED_PROPERTY)

	# Saving the resulting scores
	output_file = f'mlm_scores_TK{DEFAULT_MAX_TOKENS_NUMBER}.csv'
	output_dir = Path(f"results/{PROTECTED_PROPERTY}-{STEREOTYPED_PROPERTY}")

	output_dir.mkdir(parents=True, exist_ok=True)
	resulting_scores.to_csv(output_dir / output_file, index=False)

