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

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.parent))
from model.cross_scoring.mlm import MLMCrossScorer
from model.cross_scoring.base import CrossScorer
from model.cross_scoring.pppl import PPPLCrossScorer
from data_processing.sentence_maker import get_generation_datasets


if __name__ == '__main__':

	protected_property = 'gender'
	stereotyped_property = 'profession'
	generation_file_id = 1

	scorer: CrossScorer = PPPLCrossScorer(max_tokens_number=1, discard_longer_words=True)

	pp_words, sp_words, templates = get_generation_datasets(protected_property, stereotyped_property, generation_file_id)
	pp_words = pp_words
	sp_words = sp_words.shuffle().select(range(50))

	scores = scorer.compute_cross_scores(templates, pp_words, sp_words)
	pp_values, sp_values, avg_scores = scorer.average_by_values(*scores)

	print("Averaged scores by values: ")
	for i, pv in enumerate(pp_values):
		for j, sv in enumerate(sp_values):
			print(f"{pv} - {sv}: {avg_scores[i][j]}")