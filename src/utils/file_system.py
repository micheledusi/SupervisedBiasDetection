# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module offers some utility functions to retrieve the paths of the files
# used in the project, based on some naming conventions.

from data_processing.data_reference import BiasDataReference
from utils.config import Configurations, Parameter
from utils.const import FOLDER_RESULTS, FOLDER_CROSSED_EVALUATION, FILE_GENERATION


def get_crossed_evaluation_data_folder(protected_property: str, stereotyped_property: str) -> str:
    inner_folder: str = f"{protected_property}-{stereotyped_property}"
    return FOLDER_CROSSED_EVALUATION + "/" + inner_folder


def get_file_desinence(id: int) -> str:
    return f"-{id:02d}"


def get_crossed_evaluation_generation_file(bias_reference: BiasDataReference) -> str:
    inner_folder: str = f"{bias_reference.protected_property.name}-{bias_reference.stereotyped_property.name}"
    return FOLDER_CROSSED_EVALUATION + "/" + inner_folder + "/" + FILE_GENERATION + get_file_desinence(bias_reference.generation_id) + ".json"


def get_model_results_folder(configs: Configurations) -> str:
    return f"{FOLDER_RESULTS}/{configs[Parameter.MODEL_NAME]}"