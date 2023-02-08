# - - - - - - - - - - - - - - - #
#	Supervised Bias Detection	#
#								#
#	Author:  Michele Dusi		#
#	Date:	2023				#
#								#
#		 __TEST_FILE__ 			#
#								#
# - - - - - - - - - - - - - - - #

# This file contains the tests for the SVM classifier.


import sys
from pathlib import Path
from datasets import Dataset

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.parent))
from data_processing.sentence_maker import PP_PATTERN, SP_PATTERN
from model.classification.linear import LinearClassifier
from model.classification.svm import SVMClassifier
from utils.cache import get_cached_embeddings


if __name__ == '__main__':

	property = 'religion'
	property_type = 'protected' # "stereotyped", "protected"

	pattern = SP_PATTERN if property_type == 'stereotyped' else PP_PATTERN if property_type == 'protected' else None
	words_id = '02'
	templates_id = '00' if property_type == 'protected' else '01' if property_type == 'stereotyped' else None
	words_file = f'data/{property_type}-p/{property}/words-{words_id}.csv'
	templates_file = f'data/{property_type}-p/{property}/templates-{templates_id}.csv'
	embedding_dataset = get_cached_embeddings(property, pattern, words_file, templates_file, rebuild=False)

	def print_dataset_info(dataset: Dataset):
		# Printing the number of samples in the dataset
		print("Number of samples: ", len(dataset))
		# Printing the number of samples for each value (i.e., for each class) in the 'value' column
		classes = set(dataset['value'])
		print("Number of different classes: ", len(classes))
		# For each value (i.e., for each class) in the 'value' column, printing the number of samples
		for value in classes:
			num_samples = len(list(filter(lambda x: x == value, dataset['value'])))
			print(f"\tNumber of samples for class '{value}': ", num_samples)

	# Squeezing the embeddings to remove the template dimension and the token dimension
	def squeeze_fn(sample):
		sample['embedding'] = sample['embedding'].squeeze()
		return sample
	embedding_dataset = embedding_dataset.map(squeeze_fn, batched=True)

	# Splitting the dataset into train and test
	embedding_dataset = embedding_dataset.train_test_split(test_size=0.5, shuffle=True, seed=42)

	# Using the embeddings to train the model
	reg_model = SVMClassifier()
	# reg_model = LinearClassifier()
	reg_model.train(embedding_dataset['train'])
	print_dataset_info(embedding_dataset['train'])

	# Predict the values
	results = reg_model.evaluate(embedding_dataset['test'])

	# Print the results
	guesses = 0
	for result in results:
		predicted_value = reg_model.classes[result['prediction']]
		actual_value = result['value']
		print(f"Word: {result['word']:20s}", end=' ')
		print(f"Predicted class:", predicted_value, end='\t')
		print(f"Actual class:", actual_value, end='\t')
		if predicted_value == actual_value:
			guesses += 1
			print()
		else:
			print(" WRONG")

	print(f"Validation accuracy: {guesses}/{len(results)} ({guesses/len(results)*100:.2f}%)")

	# Print the relevance of each feature
	features_relevance = reg_model.features_relevance
	print("Features relevance dimensions: ", features_relevance.shape)