# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

from datasets import Dataset
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import sys
from pathlib import Path

directory = Path(__file__)
sys.path.append(str(directory.parent.parent))

from utility import const
from data_processing import sentence_maker
from data_processing.sentence_maker import get_generation_datasets


class MLMPredictor:
	"""
	This class contains the methods for predicting a protected property value using a masked language model.
	"""

	__softmax = torch.nn.Softmax(dim=-1)

	def __init__(self) -> None:
		self.tokenizer = AutoTokenizer.from_pretrained(const.DEFAULT_BERT_MODEL_NAME)
		self.model = AutoModelForMaskedLM.from_pretrained(const.DEFAULT_BERT_MODEL_NAME)


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


PROTECTED_PROPERTY: str = "religion"
STEREOTYPED_PROPERTY: str = "quality"

if __name__ == "__main__":
	# Retrieve the datasets
	pp_words, sp_words, templates = get_generation_datasets(PROTECTED_PROPERTY, STEREOTYPED_PROPERTY, 1)

	print("Number of protected words:", len(pp_words))
	print("Number of stereotyped words:", len(sp_words))
	print("Number of templates:", len(templates))

	model = MLMPredictor()
	resulting_scores: Dataset = Dataset.from_dict({})

	# For each stereotyped word
	for sw in tqdm(sp_words):

		# For each template, we insert the stereotyped word in the slot
		sentences_pairs = [sentence_maker.replace_stereotyped_word(sent, sw) for sent in templates['template']]
		sentences = [sent_pair[0] for sent_pair in sentences_pairs if sent_pair[1] is True]

		# Prepare the dictionary for storing the scores for each protected value
		protected_scores: dict[str, float] = {"stereotyped_word": sw['word'], "stereotype_value": sw['value']}

		for pw in pp_words:
			masked_sentences = [sentence_maker.mask_protected_word(sent) for sent in sentences]
			score: float = model.predict(masked_sentences, pw['word'])

			# We assume that a protected value is represented by only one protected word
			protected_scores[pw['value']] = score

		resulting_scores = resulting_scores.add_item(protected_scores)

	print(resulting_scores)

	# Saving the resulting scores
	output_file = 'mlm_scores.csv'
	output_dir = Path(f"results/{PROTECTED_PROPERTY}-{STEREOTYPED_PROPERTY}")

	output_dir.mkdir(parents=True, exist_ok=True)
	resulting_scores.to_csv(output_dir / output_file, index=False)

