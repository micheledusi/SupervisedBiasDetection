# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# This module contains the constants used in the project.

# Filename and folder conventions
# All paths are relative to the root of the project.
FOLDER_DATA: str = "data"
FOLDER_CROSSED_EVALUATION: str = FOLDER_DATA + "/crossed-evaluation"
FOLDER_PROTECTED_PROPERTY: str = FOLDER_DATA + "/protected-p"
FOLDER_STEREOTYPED_PROPERTY: str = FOLDER_DATA + "/stereotyped-p"

FILE_VALUES: str = "values"
FILE_WORDS: str = "words"
FILE_TEMPLATES: str = "templates"
FILE_GENERATION: str = "sentence-generation"

# Properties names
PP_GENDER: str = "gender"
PP_RELIGION: str = "religion"
SP_PROFESSION: str = "profession"
SP_QUALITY: str = "quality"
SP_ACTION: str = "action"

# Bert tokenization
TOKEN_CLS: str = "[CLS]"
TOKEN_SEP: str = "[SEP]"
TOKEN_MASK: str = "[MASK]"

# Bert model name
DEFAULT_BERT_MODEL_NAME: str = "bert-base-uncased"