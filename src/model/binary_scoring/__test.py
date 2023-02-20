# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	2023				#
#								#
#		 __TEST_FILE__ 			#
#								#
# - - - - - - - - - - - - - - - #

# This file contains the tests for the Cross-Scoring module.


import sys
from pathlib import Path

from utils.config import Configurations, Parameter

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.parent))
from model.binary_scoring.base import BinaryScorer
from data_processing.sentence_maker import get_generation_datasets


if __name__ == '__main__':

	protected_property = 'religion'
	stereotyped_property = 'criminality'
	generation_file_id = 1

	pp_words, sp_words, templates = get_generation_datasets(protected_property, stereotyped_property, generation_file_id)
	pp_words = pp_words
	sp_words = sp_words

	# Configurations
	configs = Configurations({
		Parameter.CROSSING_STRATEGY: 'pppl',
		Parameter.POLARIZATION_STRATEGY: 'difference',
		Parameter.MAX_TOKENS_NUMBER: 1,
		Parameter.DISCARD_LONGER_WORDS: True
	})

	binary_scorer = BinaryScorer(configs)
	pp_values, sp_values, polarization_scores = binary_scorer(templates, pp_words, sp_words)

	print("Polarization scores:", polarization_scores)
	print(polarization_scores.data)

	# scorer: CrossScorer = PPPLCrossScorer(max_tokens_number=1, discard_longer_words=True)
	# scores = scorer.compute_cross_scores(templates, pp_words, sp_words)
	# pp_values, sp_values, avg_scores = scorer.average_by_values(*scores)

	# print("Averaged scores by values: ")
	# for i, pv in enumerate(pp_values):
	# 	for j, sv in enumerate(sp_values):
	# 		print(f"{pv} - {sv}: {avg_scores[i][j]}")