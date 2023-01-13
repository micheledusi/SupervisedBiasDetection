# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module will be used to process the input data for the crossed evalutation.
# It will generate the sentences based on the templates and the given words.

# Jumping to parent directory with imports
import sys
from pathlib import Path

directory = Path(__file__)
sys.path.append(str(directory.parent.parent))

# Imports
import json
from datasets import Dataset
import re
from itertools import product

from utility import file_system as fs
from utility.article_inference import add_article


PROTECTED_PROPERTY: str = "gender"
STEREOTYPED_PROPERTY: str = "profession"


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
        return Dataset.from_csv(words_json['data'])

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


def get_generation_datasets(protected_property: str, stereotyped_property: str, file_id: int = 1) -> tuple[Dataset, Dataset, Dataset]:
    """
    This function will return a tuple containing the three datasets:
    - The protected property words dataset.
    - The stereotyped property words dataset.
    - The templates dataset.

    :return: The tuple containing the three datasets.
    """
    filepath: str = fs.get_crossed_evaluation_generation_file(protected_property, stereotyped_property, id=file_id)
    with open(filepath, "r") as file:
        # Obtain the generation-sentence data object
        gen_obj = json.load(file)
        pp_words: Dataset = get_dataset_from_words_json(gen_obj["protected-words"])
        sp_words: Dataset = get_dataset_from_words_json(gen_obj["stereotyped-words"])
        templates: Dataset = get_dataset_from_templates_json(gen_obj["templates"])
        return pp_words, sp_words, templates


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

                found = re.findall(pattern, sentence)
                for matched in found:
                    mask, required_descriptor = matched[0], matched[1]
                    if required_descriptor is None or required_descriptor == '':
                        # No descriptor is required, so any word can be inserted
                        sentence = sentence.replace(mask, word['word'])
                    elif 'descriptor' in word and word['descriptor'] == required_descriptor:
                        # OR the descriptor is required and it matches the word's descriptor (which exists)
                        sentence = sentence.replace(mask, word['word'])
                    elif required_descriptor == 'art':
                        # Special case: we need to add the indeterminative article
                        sentence = sentence.replace(mask, add_article(word['word']))
                    else:
                        # If the descriptor is required but it doesn't match, the word is not inserted and we skip to the next match
                        all_words_instanced = False
                        break
            
            # Check if all the words have been instanced
            # We exit the loop only if all the words have been instanced
            replaced = all_words_instanced

        # At the end, we return:
        #   - The original template string.
        #   - The word, as a structured object (dictionary) with the word, the value and the descriptor.
        #   - The instanced template string, with the protected word inserted only if the descriptor matches.
        return template['template'], words, sentence


if __name__ == "__main__":
    
    PP_PATTERN = r'(\[PROT\-PROP(?:\:([A-Za-z\-]+))?\])'
    SP_PATTERN = r'(\[STER\-PROP(?:\:([A-Za-z\-]+))?\])'
    # These pattern will extract two groups:
    #   - The first group will be the whole "protected word" mask, that is the part that's going to be replaced.
    #   - The second group will be the word descriptor, that is going to be used to select the word.


    # Retrieve the datasets
    pp_words, sp_words, templates = get_generation_datasets(PROTECTED_PROPERTY, STEREOTYPED_PROPERTY, 1)

    print("Number of protected words:", len(pp_words))
    print("Number of stereotyped words:", len(sp_words))
    print("Number of templates:", len(templates))

    count = 0
    for _, word, sentence in TemplateCombinator((SP_PATTERN, sp_words), (PP_PATTERN, pp_words), templates_dataset=templates):
        count += 1
        # TODO: Do something with the sentence
        print(sentence)
        if count > 10:
            break