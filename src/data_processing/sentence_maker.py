# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module will be used to process the input data for the crossed evalutation.
# It will generate the sentences based on the templates and the given words.

import json
from datasets import Dataset
import re
from itertools import product

from data_processing.data_reference import BiasDataReference
from data_processing.pattern import PATTERN
from utils.const import TOKEN_MASK
from utils.file_system import get_crossed_evaluation_generation_file
from utils.article_inference import add_article


def get_dataset_from_words_csv(words_csv_file: str) -> Dataset:
    dataset: Dataset = Dataset.from_csv(words_csv_file)
    if 'word' not in dataset.column_names:
        raise ValueError("The column 'word' is not present in the words dataset.")
    if 'value' not in dataset.column_names:
        # If no value is present, we copy the word column
        dataset = dataset.add_column('value', dataset['word'])
    return dataset


def get_dataset_from_words_json(words_json: dict) -> Dataset:
    """
    This function will return a Dataset object from the given JSON object.
    The JSON object can have two types of data: a file path or an "array".
    If the type is "file", the function will return a Dataset object from the CSV file.
    If the type is "array", the function will return a Dataset object from a structured array.

    :param words_json: The JSON object containing the data.
    :return: The Dataset object.
    """
    # Case 1: the type is "file"
    if words_json['type'] == 'file':
        return get_dataset_from_words_csv(words_json['data'])

    # Case 2: the type is "array"
    elif words_json['type'] == 'array':
        dataset_dict: dict[str, list[str]] = {'word': [], 'value': [], 'descriptor': []}
        for row in words_json['data']:
            for key in ['word', 'value', 'descriptor']:
                if key not in row:
                    raise ValueError(f"The key {key} is not present in the row {row}.")
                dataset_dict[key].append(row[key])
        return Dataset.from_dict(dataset_dict)
    
    # Default case: if the type is not valid, raise an error
    else:
        raise ValueError("Invalid type of data for the words.")


def get_dataset_from_templates_json(templates_json: dict) -> Dataset:
    # Case 1: the type is "file"
    if templates_json['type'] == 'file':
        return Dataset.from_csv(templates_json['data'])

    # Case 2: the type is "array"
    elif templates_json['type'] == 'array':
        raise NotImplementedError("The array type for the templates is not implemented yet.")

    # Default case: if the type is not valid, raise an error
    raise ValueError("Invalid type of data for the templates.")


def get_generation_datasets(bias_reference: BiasDataReference) -> tuple[Dataset, Dataset, Dataset]:
    """
    This function will return a tuple containing the three datasets:
    - The protected property words dataset.
    - The stereotyped property words dataset.
    - The templates dataset.

    :return: The tuple containing the three datasets.
    """
    filepath: str = get_crossed_evaluation_generation_file(bias_reference)
    with open(filepath, "r") as file:
        # Obtain the generation-sentence data object
        gen_obj = json.load(file)
        pp_words: Dataset = get_dataset_from_words_json(gen_obj["protected-words"])
        sp_words: Dataset = get_dataset_from_words_json(gen_obj["stereotyped-words"])
        templates: Dataset = get_dataset_from_templates_json(gen_obj["templates"])
        return pp_words, sp_words, templates


def replace_word(sentence: str, word_sample: dict[str, str], pattern: re.Pattern[str]) -> tuple[str, bool]:
    """
    This function will replace all the occurrences of the given pattern in the given sentence with the given word.
    If the sentence contain multiple occurrences of the pattern, each one will be replaced with the given word
    if and only if the word's descriptor matches the required descriptor (if any).

    A special case is when the required descriptor is None or an empty string. In this case, the word can be inserted in any case.
    Another special case is when the required descriptor is "art", in which case the word will be inserted with the correct indeterminative article
    (e.g. "a" or "an", inferred from the word itself).

    The method considers the replacement successful if at least one replacement was made, and thus it returns True and the modified sentence.
    If no replacement was made, the method returns False (along with the unmodified sentence).

    :param sentence: The sentence in which the word will be inserted.
    :param word: The word that will be inserted.
    :param pattern: The pattern that will be used to find the word.
    :return: The sentence with the word inserted and a boolean indicating if the word was inserted at least once (True) or not (False).
    """
    assert 'word' in word_sample, "The word must have a 'word' key."
    replaced: bool = False
    found = pattern.findall(sentence)
    for matched in found:
        mask, required_descriptor = matched[0], matched[1]
        if not required_descriptor:
            # No descriptor is required, so any word can be inserted
            sentence = sentence.replace(mask, word_sample['word'])
            replaced = True
        elif word_sample.get('descriptor', '') == required_descriptor:
            # OR the descriptor is required and it matches the word's descriptor (which exists)
            sentence = sentence.replace(mask, word_sample['word'])
            replaced = True
        elif required_descriptor == 'art':
            # Special case: we need to add the indeterminative article
            sentence = sentence.replace(mask, add_article(word_sample['word']))
            replaced = True
    return sentence, replaced


def replace_word_in_templates(word_sample: dict[str, str], templates: Dataset, pattern: re.Pattern[str]) -> Dataset:
    """
    This function will replace all the occurrences of the given pattern in the given templates with the given word.
    If a template contain multiple occurrences of the pattern, each one will be replaced with the given word
    if and only if the word's descriptor matches the required descriptor (if any).

    A special case is when the required descriptor is None or an empty string. In this case, the word can be inserted in any case.
    Another special case is when the required descriptor is "art", in which case the word will be inserted with the correct indeterminative article
    (e.g. "a" or "an", inferred from the word itself).

    At the end, only the templates with at least one replacement will be returned.

    :param word_sample: The word that will be inserted, as a row of the <words> dataset.
    :param templates: The templates in which the word will be inserted, as a dataset.
    :param pattern: The pattern that will be used to find the word.
    :return: A dataset containing the templates ('template' column) and the formed sentences ('sentence' column), then filtered for only the correct replacements.
    """
    assert 'template' in templates.column_names, "The templates must have a 'template' column."
    # For each template, we replace the word in the pattern through the `replace_word` function
    sentences_tuples = [replace_word(template, word_sample, pattern) for template in templates['template']]
    # Adding the columns to the dataset
    result_ds: Dataset = templates.add_column('sentence', [tpl[0] for tpl in sentences_tuples])
    result_ds = result_ds.add_column('replacement', [tpl[1] for tpl in sentences_tuples])
    # We filter the sentences to keep only the ones with the word inserted
    result_ds = result_ds.filter(lambda x: x['replacement'] == True)
    # We keep only the sentences, without the boolean flag
    return result_ds.remove_columns('replacement')


def mask_word(sentence: str, pattern: str, word_sample: dict[str, str] = None) -> tuple[str, bool]:
    """
    This function will mask all the occurrences of the given pattern in the given sentence.
    If the sentence contain multiple occurrences of the pattern, each one will be masked.

    If the word is not None, the method will also check if the word's descriptor matches the required descriptor (if any).
    If the given word is None, the method will blindly mask all the occurrences of the pattern.

    :param sentence: The sentence in which the word will be masked.
    :param word_sample: The word that constraints the masking, if the word's descriptor matches the required descriptor in the sentence.
    :param pattern: The pattern that will be used to find the word.
    :return: The sentence with the word masked.
    """
    # Create a masked word
    masked_word: dict[str, str] = word_sample.copy() if word_sample is not None else {}
    masked_word['word'] = TOKEN_MASK

    # Replace the pattern with the masked word, only if the word's descriptor matches the required descriptor (if any)
    return replace_word(sentence, masked_word, pattern)


class TemplateCombinator:
    """
    This class is used to iterate over two lists of, respectively, templates and words.
    The number of iterations will be the product of the number of templates and the number of words.
    Each iteration will return a tuple containing:
    - The original template string.
    - The word, as a structured object (dictionary) with the word, the value and the descriptor.
    - The instanced template string, with the protected word inserted only if the descriptor matches.
    """
    def __init__(self, *args, templates_dataset: Dataset):
        self.templates = templates_dataset
        self.pattern_list: list[str] = [pair[0] for pair in args]
        words_list: list[Dataset] = [pair[1] for pair in args]
        self.combinator = product(self.templates, *words_list)

    def __iter__(self):
        return self

    def __next__(self) -> tuple:
        replaced = False
        while (not replaced):
            # The "combination" part is handled by the product function
            current_combination = next(self.combinator)
            template = current_combination[0]
            words = current_combination[1:]
            sentence: str = template['template']

            all_words_instanced: bool = True
            for pattern, word in zip(self.pattern_list, words):
                sentence, replacement_done = replace_word(sentence, word, pattern)

                # If the replacement was not done, we set the flag to False and we exit the loop
                if not replacement_done:
                    all_words_instanced = False
                    break
                
                # TODO: Al momento c'è ancora la possibilità che un template abbia più "slot" con lo stesso pattern.
                #       In questo caso, se una delle sostituzioni non va a buon fine, il metodo di sostituzione restituisce comunque "True",
                #       perché almeno una delle sostituzioni è andata a buon fine.
                #       Al momento assumiamo quindi che ogni template abbia un solo "slot" per ogni pattern.
            
            # Check if all the words have been instanced
            # We exit the loop only if all the words have been instanced
            replaced = all_words_instanced

        # At the end, we return:
        #   - The original template string.
        #   - The word, as a structured object (dictionary) with the word, the value and the descriptor.
        #   - The instanced template string, with the protected word inserted only if the descriptor matches.
        return template['template'], words, sentence


def template_combinator(templates_dataset: Dataset, stereotyped_words: Dataset, protected_words: Dataset) -> TemplateCombinator:
    return TemplateCombinator((PATTERN, stereotyped_words), (PATTERN, protected_words), templates_dataset=templates_dataset)
