# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module contains a class used to extract embeddings from a word.
# The class has a core method, called "embed", which takes a word as input (along with some other inputs) and returns the embedding of the word.
# The embedding is a vector of floats, instanced as a PyTorch tensor.
# This process can be done in different ways, depending on what the user wants to do.

import torch
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer

import sys
from pathlib import Path

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.parent))
from data_processing.sentence_maker import SP_PATTERN, replace_word
from utility.const import DEFAULT_BERT_MODEL_NAME, TOKEN_CLS, TOKEN_SEP

EMPTY_TEMPLATE = TOKEN_CLS + ' ' + SP_PATTERN + ' ' + TOKEN_SEP


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

            - ``select_templates``: the number of templates to select for each word. 
            If the value is "-1", all the templates will be selected. The selection is done randomly. 
            By default, it will select all the templates.

            - ``average_templates``: whether to average the embeddings of each template or not, for a single word.
            By default, it will average the embeddings (i.e. the value is True).

            - ``average_tokens``: whether to average the embeddings of each token or not, for a single word.
            Note: we don't know in advance if the word will be split into multiple tokens.
            By default, it will average the embeddings (i.e. the value is True).

        This parameters are used in the "embed" method.

        :param kwargs: The keyword arguments to be passed to the embedding method.
        """
        # Processing arguments
        # The pattern used to find the word in the sentence.
        if 'pattern' in kwargs:
            self.pattern: str = kwargs['pattern']
        else:
            self.pattern: str = SP_PATTERN

        # The number of templates to select for each word.
        # If the value is "-1", all the templates will be selected.
        # The selection is done randomly.
        if 'select_templates' in kwargs:
            arg = kwargs['select_templates']
            if arg == 'all':
                self.select_templates: int = -1 # "-1" means all templates will be selected
            else:
                self.select_templates: int = max(1, arg)    # At least one template will be selected
        else:
            self.select_templates: int = -1
        
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
        #   - if the number of tokens is less or equal than the maximum number of tokens, all the tokens will be considered (and eventually averaged, see "average_tokens")
        #   - if the number of tokens is greater than the maximum number of tokens:
        #       - if the value of "discard_longer_words" is True, the word will be discarded
        #       - if the value of "discard_longer_words" is False, the considered tokens will be the first "max_tokens_number" tokens
        if 'max_tokens_number' in kwargs:
            arg = kwargs['max_tokens_number']
            if arg == 'all':
                self.max_tokens_number: int = -1 # "-1" means all tokens will be considered
            else:
                self.max_tokens_number: int = max(1, arg)    # At least one token has to be considered
        else:
            self.max_tokens_number: int = -1
        
        # Whether to discard the words that are split into more tokens than the maximum number of tokens
        if 'discard_longer_words' in kwargs:
            self.discard_longer_words = kwargs['discard_longer_words']
        else:
            self.discard_longer_words = False

        # The model used to extract the embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BERT_MODEL_NAME)
        self.model = AutoModel.from_pretrained(DEFAULT_BERT_MODEL_NAME)

    def _get_subsequence_index(self, array: torch.Tensor, subarray: torch.Tensor) -> torch.Tensor:
        # Example:
        # array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # subarray = [3, 4, 5]
        # output = [2]
        window_len = subarray.shape[-1]
        steps = array.shape[-1] - window_len + 1
        # Unfold the last dimension of the array into 2 dimension of length [len(array) - window_len + 1, window_len]
        unfolded_array = array.unfold(dimension=-1, size=window_len, step=1)
        # print("Unfolded array shape:", unfolded_array.shape)
        # Repeat the subarray to match the shape of the unfolded array
        repeated_subarray = subarray.unsqueeze(0).repeat(steps, 1)
        # print("Repeated subarray shape:", repeated_subarray.shape)
        # Both arrays have the same shape now
        # Shape = [#sentences_padded_tokens, #word_tokens]
        # Compare the two arrays:
        comparison = torch.all(unfolded_array == repeated_subarray, dim=-1)
        # print("Comparison shape:", comparison.shape)
        first_occurrence_index = int(torch.where(comparison == True)[0])
        # We get to a single scalar
        # Now we repeat the first occurrence index (increasing it) for each element of the subarray
        # Shape = [#word_tokens]
        return torch.arange(start=first_occurrence_index, end=first_occurrence_index + window_len, dtype=torch.long)


    def embed_word(self, word: dict[str, str], templates: Dataset) -> torch.Tensor:
        """
        This method takes a word and returns its embedding(s).

        In order to do so, it uses a dataset of templates. The templates are sentences that can contain the word we want to embed.

        The resulting embedding is a PyTorch tensor of size [#templates, #tokens, #features].
        Note that:
        - If the ``average_templates`` parameter is True, the resulting embedding will be of size [1, #tokens, #features].
        - If the ``average_tokens`` parameter is True, the resulting embedding will be of size [#templates, 1, #features].
        - For standard BERT models, #features is 768, thus the resulting embedding will be of size [#templates, #tokens, 768].

        :param word: The word to be embedded, as a dictionary with the following keys: "word", "descriptor", and (optional) "value".
        :param kwargs: The keyword arguments to be passed to the embedding method.
        :return: The embedding of the word (or None if the word is split into more tokens than the maximum number of tokens and the ``discard_longer_words`` parameter is True)
        """
        # Tokenizing the word
        word_tokens = self.tokenizer(word['word'], padding=False, truncation=False, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
        # Note: the special tokens [CLS] and [SEP] are not considered (i.e. they are not added to the tokenized word)
        # If the word is split into more tokens than the maximum number of tokens, we either discard it or truncate it
        if self.max_tokens_number != -1 and len(word_tokens) > self.max_tokens_number:
            if self.discard_longer_words:
                return None
            else:
                word_tokens = word_tokens[:self.max_tokens_number]

        # Replacing the word in the templates, with a map function
        def replace_word_fn(sample):
            sentence, replacement = replace_word(sentence=sample['template'], word=word, pattern=self.pattern)
            sample['sentence'] = sentence
            sample['replacement'] = replacement
            return sample
        sentences = templates.map(replace_word_fn, batched=False)

        # Then selecting only the templates where the word was replaced
        # And finally, selecting a random subset of templates if needed
        sentences = sentences.filter(lambda x: x['replacement'] == True)
        if self.select_templates == -1 or self.select_templates >= len(sentences):
            sentences = sentences
        else:
            sentences = sentences.shuffle().select(range(self.select_templates))

        # Tokenizing the sentences
        tokenized_sentences = self.tokenizer(sentences['sentence'], padding=True, truncation=True, return_tensors='pt')

        # Tokenized inputs
        sentences_tokens = tokenized_sentences['input_ids']  # The sentences tokens
        # print("Sentences tokens size:", sentences_tokens.shape)

        # Getting the indices of the word tokens in the sentences tokens
        word_tokens_indices = torch.stack([self._get_subsequence_index(sent_tokens, word_tokens) for sent_tokens in sentences_tokens])
        # print("Word tokens indices:", word_tokens_indices)

        # Embedding the templates
        embeddings = self.model(sentences_tokens)
        embeddings = embeddings['last_hidden_state']
        # print("Embeddings shape: ", embeddings.size())

        # Stacking and aggregating the embeddings
        word_embeddings = torch.stack([
            embeddings[i, word_tokens_indices[i]]
            for i in range(len(embeddings))])
        # print("Word embeddings shape: ", word_embeddings.size())
        # We're now left with a tensor of size [#templates, #tokens, #features]

        # Averaging the embeddings
        if self.average_templates:
            word_embeddings = torch.mean(word_embeddings, dim=0).unsqueeze(0)
        if self.average_tokens:
            word_embeddings = torch.mean(word_embeddings, dim=1).unsqueeze(1)

        return word_embeddings

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
        def tokenize_word_fn(sample):
            word_tokens = self.tokenizer(sample['word'], padding=False, truncation=False, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
            # Note: the special tokens [CLS] and [SEP] are not considered (i.e. they are not added to the tokenized word)
            # The 'word_tokens' is a tensor of size [#word_tokens]
            sample['tokens'] = word_tokens
            sample['num_tokens'] = word_tokens.shape[0]
            return sample

        def embed_word_fn(sample):
            embedding: torch.Tensor = self.embed_word(sample, templates)
            sample['embedding'] = embedding     # This is not None, since the longer words have been filtered out
            return sample

        # Tokenizing the words
        words = words.map(tokenize_word_fn, batched=False, num_proc=4)
        # Filtering the words that are too long
        if self.discard_longer_words and self.max_tokens_number != -1:
            print("Discarding words longer than", self.max_tokens_number, "tokens")
            words = words.filter(lambda sample: sample['num_tokens'] <= self.max_tokens_number)
        print("Number of words to be embedded:", len(words))

        # Embedding the words
        embeddings = words.map(embed_word_fn, batched=False, num_proc=4, remove_columns=['tokens', 'num_tokens'])
        # NOTE: the 'tokens' and 'num_tokens' columns are removed, since they are not needed anymore. If you want to keep them, you can edit the previous line.
        embeddings = embeddings.with_format('torch')
        return embeddings


if __name__ == "__main__":
    # Loading the datasets
    templates: Dataset = Dataset.from_csv('data/stereotyped-p/profession/templates-01.csv')
    words: Dataset = Dataset.from_csv('data/stereotyped-p/profession/words-01.csv').select(range(30, 40))

    # Creating the word embedder
    word_embedder = WordEmbedder(select_templates='all', average_templates=False, average_tokens=False, discard_longer_words=True, max_tokens_number=2)

    # Embedding a word
    embedding_dataset = word_embedder.embed(words, templates)
    print(f"Resulting embedding length:", len(embedding_dataset))
