# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module offers some utility functions to retrieve the paths of the files
# used in the project, based on some naming conventions.

from utility import const


def get_crossed_evaluation_data_folder(protected_property: str, stereotyped_property: str) -> str:
    inner_folder: str = f"{protected_property}-{stereotyped_property}"
    return const.FOLDER_CROSSED_EVALUATION + "/" + inner_folder


def get_protected_property_data_folder(protected_property: str) -> str:
    return const.FOLDER_PROTECTED_PROPERTY + "/" + protected_property


def get_stereotyped_property_data_folder(stereotyped_property: str) -> str:
    return const.FOLDER_STEREOTYPED_PROPERTY + "/" + stereotyped_property


def get_file_desinence(id: int) -> str:
    return f"-{id:02d}"


def get_crossed_evaluation_generation_file(protected_property: str, stereotyped_property: str, id: int = 1) -> str:
    inner_folder: str = f"{protected_property}-{stereotyped_property}"
    return const.FOLDER_CROSSED_EVALUATION + "/" + inner_folder + "/" + const.FILE_GENERATION + get_file_desinence(id) + ".json"