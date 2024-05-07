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
FOLDER_RESULTS: str = "results"
FOLDER_CROSSED_EVALUATION: str = FOLDER_DATA + "/crossed-evaluation"

FILE_VALUES: str = "values"
FILE_WORDS: str = "words"
FILE_TEMPLATES: str = "templates"
FILE_GENERATION: str = "sentence-generation"

# Bert tokenization
TOKEN_CLS: str = "[CLS]"
TOKEN_SEP: str = "[SEP]"
TOKEN_MASK: str = "[MASK]"

# Bert model name
MODEL_NAME_BERT_BASE_UNCASED: str = "bert-base-uncased"
MODEL_NAME_ROBERTA_BASE: str = "roberta-base"
MODEL_NAME_DISTILBERT_BASE_UNCASED: str = "distilbert-base-uncased"
MODEL_NAME_ELECTRA_BASE: str = "google/electra-base-discriminator"
DEFAULT_MODEL_NAME: str = MODEL_NAME_BERT_BASE_UNCASED

# Configurations
NUM_PROC: int = 4
# DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE: str = "cpu"
BATCH_SIZE = 32

# Embedding configurations - Raw embeddings computation
DEFAULT_LONGER_WORD_POLICY = 'truncate' # 'truncate' or 'discard' or 'ignore'
DEFAULT_MAX_TOKENS_NUMBER = 'all'
# Embedding configurations - Templates and words sampling / Embeddings combination
DEFAULT_TEMPLATES_SAMPLING_NUMBER = 3
DEFAULT_WORDS_SAMPLING_NUMBER = 30
DEFAULT_TEMPLATES_POLICY = 'average'
DEFAULT_MAX_TESTCASE_NUMBER = 100
# Embedding configurations - Testcase post-processing
DEFAULT_CENTER_EMBEDDINGS = False

DEFAULT_TEST_SPLIT_PERCENTAGE = 0.5

# Reduction configurations
DEFAULT_REDUCTION_DROPOUT_PERCENTAGE = 0.5
DEFAULT_RELEVANCE_COMPUTATION_STRATEGY = 'from_classifier' # 'from_classifier' or 'shap'
DEFAULT_CLASSIFIER_TYPE = 'svm' # 'svm', 'linear', 'tree', or 'randomforest'
DEFAULT_RELEVANCE_NORMALIZATION_STRATEGY = 'linear' # 'linear', 'linear-opposite', 'quadratic', 'quadratic-opposite', 'sigmoid', 'sigmoid-opposite', 'sigmoid-adaptive
DEFAULT_REDUCTION_TYPE = 'pca' # 'pca' or 'tsne'
DEFAULT_BIAS_TEST = 'chi2' # only 'chi2' for now

# Crossed evaluation configurations
DEFAULT_CROSS_PROBABILITY_STRATEGY = 'pppl'	# 'pppl' or 'mlm'
DEFAULT_POLARIZATION_STRATEGY = 'ratio'		# 'difference' or 'ratio'

# Separation configurations
DEFAULT_EMBEDDINGS_DISTANCE_STRATEGY = 'euclidean'	# 'euclidean' or 'cosine'

# Columns names in the datasets
COL_WORD: str = "word"
COL_CLASS: str = "value"
COL_EMBS: str = "embedding"

# ANSI Escape codes
import re
ANSI_FILTER = re.compile(r'\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?(;[0-9]{3})?)?[m|K]?')