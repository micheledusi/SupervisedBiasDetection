# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import sys
from pathlib import Path

directory = Path(__file__)
sys.path.append(str(directory.parent.parent))

from utility import const
from data_processing.sentence_maker import replace_stereotyped_word, mask_protected_word
from data_processing.sentence_maker import get_generation_datasets
from model.embedding.word_embedder import WordEmbedder


class MLMPredictor:
	"""
	This class contains the methods for predicting a protected property value using a masked language model.
	"""

	__softmax = torch.nn.Softmax(dim=-1)

	def __init__(self) -> None:
		self.tokenizer = AutoTokenizer.from_pretrained(const.DEFAULT_BERT_MODEL_NAME)
		self.model = AutoModelForMaskedLM.from_pretrained(const.DEFAULT_BERT_MODEL_NAME)
		if torch.cuda.is_available():
			self.model.cuda()

	def predict(self, sentences: str | list[str], target: str) -> float:
		"""
		This method predicts the probability that the target word is inserted in the token MASK slot.

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
			if const.TOKEN_MASK not in sentence:
				raise Exception(f"The sentence '{sentence}' does not contain the mask token ({const.TOKEN_MASK}).")
			
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

		vocabulary_scores = torch.mean(mask_token_scores, dim=0)
		# The shape of the tensor is (vocabulary_size)

		# Getting the index of the target word
		target_index = self.tokenizer.convert_tokens_to_ids(target)
		return vocabulary_scores[target_index].item()

	def compute_scores(self, protected_property: str, stereotyped_property: str, generation_file_id: int = 1) -> Dataset:
		# Retrieve the datasets for MLM
		pp_words, sp_words, templates = get_generation_datasets(protected_property, stereotyped_property, generation_file_id)

		# Prepare the dataset for storing the scores
		resulting_scores: Dataset = Dataset.from_dict({})

		# Preparing an auxiliary embedder that can be used to discard long words (i.e. words that are tokenized in more than X tokens)
		embedder = WordEmbedder()
		if const.DEFAULT_DISCARD_LONGER_WORDS:
			print("Filtering the stereotyped words...")
			sp_words = sp_words.filter(lambda x: embedder.get_tokens_number(x['word']) <= const.DEFAULT_MAX_TOKENS_NUMBER)
		
		# For each stereotyped word
		print("Computing the scores...")
		for sw in tqdm(sp_words):

			# For each template, we insert the stereotyped word in the slot
			sentences_pairs = [replace_stereotyped_word(sent, sw) for sent in templates['template']]
			sentences = [sent_pair[0] for sent_pair in sentences_pairs if sent_pair[1] is True]

			# Prepare the dictionary for storing the scores for each protected value
			protected_scores: dict[str, float] = {"stereotyped_word": sw['word'], "stereotype_value": sw['value']}

			for pw in pp_words:
				# For each protected word, we mask it in the sentences
				masked_sentences = [mask_protected_word(sent) for sent in sentences]
				# Then, we predict the probability that the protected word is inserted in the MASK slot
				protected_scores[pw['value']] = self.predict(masked_sentences, pw['word'])

			resulting_scores = resulting_scores.add_item(protected_scores)
		# Formatting with PyTorch
		resulting_scores = resulting_scores.with_format("torch", device=const.DEVICE)

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
	output_file = 'mlm_scores.csv'
	output_dir = Path(f"results/{PROTECTED_PROPERTY}-{STEREOTYPED_PROPERTY}")

	output_dir.mkdir(parents=True, exist_ok=True)
	resulting_scores.to_csv(output_dir / output_file, index=False)

