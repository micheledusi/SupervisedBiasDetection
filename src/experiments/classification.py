# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# EXPERIMENT: Dimensionality reduction over word embeddings obtained by BERT.
# DATE: 2023-05-04

from datasets import Dataset, DatasetDict
import datasets
from sklearn.metrics import accuracy_score, f1_score
import torch
from tqdm import tqdm

from data_processing.data_reference import PropertyDataReference
from experiments.base import Experiment
from model.classification.base import AbstractClassifier
from model.classification.factory import ClassifierFactory
from utils.config import ConfigurationsGrid, Parameter

configurations = ConfigurationsGrid({
	Parameter.MAX_TOKENS_NUMBER: 'all',
	Parameter.TEMPLATES_SELECTED_NUMBER: 'all',
	Parameter.CLASSIFIER_TYPE: ['svm', 'linear', 'tree'],
	Parameter.CENTER_EMBEDDINGS: False,
	Parameter.TEST_SPLIT_PERCENTAGE: [0.2],
})

TESTCASES_NUMBER: int = 100

class ClassificationExperiment(Experiment):
	"""
	In this experiment, we train a classifier on the embeddings of the protected property.
	Then, we simply measure its accuracy and F1 score on the test-set.
	"""

	def __init__(self) -> None:
		super().__init__("classification", required_kwargs=['prot_prop'])
	
	def _execute(self, **kwargs) -> None:

		datasets.disable_caching()
		datasets.disable_progress_bar()

		table_strings: list[str] = []

		for configs in configurations:
			embeddings: Dataset = Experiment._get_property_embeddings(self.protected_property, configs)
			embeddings = embeddings.add_column('labeled_value', embeddings['value'])
			embeddings = embeddings.class_encode_column('labeled_value')
			classes = embeddings.features['labeled_value'].names

			train_sizes: list[int] = []
			test_sizes: list[int] = []
			accuracies: list[float] = []
			f1s: list[float] = []

			relevances: torch.Tensor = torch.zeros((TESTCASES_NUMBER, 768))

			for testcase in tqdm(range(TESTCASES_NUMBER)):
				
				classifier: AbstractClassifier = ClassifierFactory.create(configs)

				if configs[Parameter.TEST_SPLIT_PERCENTAGE] == 0:
					classifier.train(embeddings)
				else:
					test_size_percentage = configs[Parameter.TEST_SPLIT_PERCENTAGE]
					embeddings_dict: DatasetDict = embeddings.train_test_split(
						test_size=test_size_percentage, 
						stratify_by_column='labeled_value',
						seed=testcase, 
						load_from_cache_file=False)

					classifier.train(embeddings_dict['train'])

				classifications = classifier.evaluate(embeddings_dict['test'])
				classifications = classifier.prediction_to_value(classifications)

				accuracy = accuracy_score(classifications['value'], classifications['predicted_value'])
				f1 = f1_score(classifications['value'], classifications['predicted_value'], average='macro')

				train_sizes.append(len(embeddings_dict['train']))
				test_sizes.append(len(embeddings_dict['test']))
				accuracies.append(accuracy)
				f1s.append(f1)

				relevances[testcase] = classifier.features_relevance

			avg_relevance = torch.mean(relevances, dim=0)
			std_relevance = torch.std(relevances, dim=0)
			# print(avg_relevance)
			# print(std_relevance)

			avg_train_size = sum(train_sizes) / TESTCASES_NUMBER
			avg_test_size = sum(test_sizes) / TESTCASES_NUMBER
			avg_accuracy = sum(accuracies) / TESTCASES_NUMBER
			avg_f1 = sum(f1s) / TESTCASES_NUMBER

			print("\n", configs.to_strdict())
			print("Number of classes:", len(classes), classes)
			print("Average train size:", avg_train_size)
			print("Average test size:", avg_test_size)
			print("Average accuracy score:", avg_accuracy)
			print("Average F1 score:", avg_f1)

			s = f" & {avg_accuracy:.3f}"
			table_strings.append(s)
			# print("\n\nRelevance of each feature:")
			# for i, w in enumerate(avg_relevance):
			#	print(f"{i}, {w:.8f}, {std_relevance[i]:.4f}")

		print("\n", self.protected_property.name,"\n")
		print(f" & & {100 - test_size_percentage * 100:.0f}\\% ({avg_train_size:.0f}) & {test_size_percentage * 100:.0f}\\% ({avg_test_size:.0f})", end="")
		for string in table_strings:
			print(string, end=" ")
		print("\\\\")