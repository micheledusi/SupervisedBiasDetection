# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#                               #
#   Author:  Michele Dusi       #
#   Date:    2023               #
# - - - - - - - - - - - - - - - #

# Patterns for protected and stereotyped words.

import re


PP_PATTERN = r'(\[PROT\-WORD(?:\:([A-Za-z\-]+))?\])'
PP_PATTERN: re.Pattern[str] = re.compile(PP_PATTERN)

SP_PATTERN = r'(\[STER\-WORD(?:\:([A-Za-z\-]+))?\])'
SP_PATTERN: re.Pattern[str] = re.compile(SP_PATTERN)
# These pattern will extract two groups:
#   - The first group will be the whole "protected word" mask, that is the part that's going to be replaced.
#   - The second group will be the word descriptor, that is going to be used to select the word.

PATTERN = r'(\[WORD(?:\:([A-Za-z\-]+))?\])'
PATTERN: re.Pattern[str] = re.compile(PATTERN)
# This pattern is the generalization of the previous two patterns.