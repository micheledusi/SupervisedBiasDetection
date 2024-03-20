# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module contains the class "EmbeddingPlotter" which is used to plot the embeddings.

from typing import Any
from datasets import Dataset
import matplotlib.pyplot as plt
import torch

from utils.const import COL_CLASS, COL_EMBS, COL_WORD

# Type aliases
SplitDataset = dict[Any, Dataset]

# Constants
MARKERS: list = ['v', (5, 1), 's', '<', '>', 'v', 'p', 'P', '*']
COLORS: list = ['#FD38', '#36F8', '#D3A8', 'c', 'm', 'y', 'k', 'w']

DEFAULT_COLUMN_X = "x"
DEFAULT_COLUMN_Y = "y"


class ScatterPlotter:
    """
    This class is used to plot the embeddings.
    The only requirement is that the dataset contains the columns "x" and "y".
    """

    def __init__(self, data: Dataset, title: str = None, 
                 x_col: str = DEFAULT_COLUMN_X, y_col: str = DEFAULT_COLUMN_Y, 
                 label_col: str = COL_WORD, color_col: str = COL_CLASS) -> None:
        """
        The initializer for the ScatterPlotter class.

        :param data: The dataset containing the embeddings.
        :param title: The title of the plot.
        """
        self._title = title
        self._x = data[x_col]
        self._y = data[y_col]
        self._labels = data[label_col] if label_col in data.column_names else []

        # Check if the data type of the color column is int
        colortype = 'int'
        for i in range(len(data)):
            if not isinstance(data[color_col][i], int):
                colortype = 'symbolic'
                break

        if colortype == 'int':
            self._colors = data[color_col]
        else:
            kvals: dict[str, int] = {}
            self._colors = []
            for color_sample in data[color_col]:
                if color_sample not in kvals:
                    kvals[color_sample] = len(kvals)
                self._colors.append(kvals[color_sample])


    def show(self) -> None:
        """
        This method shows the plot.
        """
        # Plotting
        if not self._colors:
            print("No colors to show.")
            plt.scatter(self._x, self._y)
        else:
            plt.scatter(self._x, self._y, c=self._colors)

        # Adding labels
        if not self._labels:
            print("No labels to show.")
        else:
            for i, txt in enumerate(self._labels):
                plt.annotate(txt, (self._x[i], self._y[i]))

        # Adding title
        if self._title is not None:
            plt.title(self._title)

        # Showing
        plt.show()



class DatasetPairScatterPlotter:
    """
    This class is used to plot the embeddings from the Protected and Stereotyped datasets.
    Both datasets have to be provided in the initializer.
    The object would assume that the datasets contain the "embeddings" column, of size 2.
    """

    def __init__(self, protected_ds: Dataset, stereotyped_ds: Dataset,
                 title: str = None, use_latex: bool = True) -> None:
        """
        The initializer for the DatasetPairScatterPlotter class.

        :param prot_ds: The dataset containing the embeddings of the protected words.
        :param ster_ds: The dataset containing the embeddings of the stereotyped words.
        :param coordinates_column: The name of the column containing the embeddings. Default is "embedding".
        :param title: The title of the plot.
        """
        self._title = title
        self._prot_ds = protected_ds
        self._ster_ds = stereotyped_ds

        # if use_latex:
        #     plt.rc('text', usetex=True)
        #     plt.rc('font', family='serif')


    def show(self) -> None:
        """
        This method shows the plot.
        """
        plt.figure(figsize=(8, 6))

        # PROTECTED DATASET
        # Splitting the dataset by class
        prot_ds_by_class: SplitDataset = DatasetPairScatterPlotter._split_dataset_by_class(self._prot_ds)
        marker_dict: dict[str, str] = DatasetPairScatterPlotter._zip_classes_to_values(prot_ds_by_class, MARKERS)

        # Extracting the coordinates for each partition dataset
        prot_xs, prot_ys = DatasetPairScatterPlotter._extract_coordinates(prot_ds_by_class)

        # Plotting
        for class_label in prot_ds_by_class:
            plt.scatter(prot_xs[class_label], prot_ys[class_label], c='white', marker=marker_dict[class_label], edgecolor='black', linewidth=0.8, label=class_label)

        # STEREOTYPED DATASET
        # Splitting the dataset by class
        ster_ds_by_class: SplitDataset = DatasetPairScatterPlotter._split_dataset_by_class(self._ster_ds)
        color_dict: dict[str, str] = DatasetPairScatterPlotter._zip_classes_to_values(ster_ds_by_class, COLORS)

        # Extracting the coordinates for each partition dataset
        ster_xs, ster_ys = DatasetPairScatterPlotter._extract_coordinates(ster_ds_by_class)

        # Plotting
        for class_label in ster_ds_by_class:
            plt.scatter(ster_xs[class_label], ster_ys[class_label], c=color_dict[class_label], label=class_label)

        # Adding labels
        DatasetPairScatterPlotter._annotate_words(self._prot_ds)
        # DatasetPairScatterPlotter._annotate_words(self._ster_ds)

        # Adding title
        if self._title is not None:
            plt.title(self._title)

        # Showing
        plt.legend()
        plt.show()


    def _annotate_words(ds: Dataset, word_col: str = COL_WORD):
        xs, ys = DatasetPairScatterPlotter._extract_coordinates(ds)
        for word, x, y in zip(ds[word_col], xs, ys):
            plt.annotate(word, xy=(x, y))


    @staticmethod
    def _split_dataset_by_class(ds: Dataset, col_class: str = COL_CLASS) -> SplitDataset:
        """
        This method splits the dataset by the values of the class column.
        It returns a dictionary containing the datasets split by class.

        :param ds: The dataset to split.
        :param col_class: The name of the column containing the class label.
        :return: A dictionary containing the datasets split by class.
        """
        class_indices: dict = {}
        for class_label in ds[col_class]:
            if class_label not in class_indices:
                class_indices[class_label] = len(class_indices)
        class_dict: dict = {class_label: ds.filter(lambda x: x[col_class] == class_label) for class_label in class_indices}
        return class_dict
    

    @staticmethod
    def _extract_coordinates(class_ds: SplitDataset | Dataset, col_embs: str = COL_EMBS) -> tuple[dict[Any, torch.Tensor], dict[Any, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor]:
        """
        This method extracts the coordinates from the dataset.
        It returns a tuple containing the x and y coordinates, if the dataset is unique,
        or a pair of dictionaries containing the coordinates for each class.

        :param class_ds: The dataset to extract the coordinates from.
        :param col_embs: The name of the column containing the embeddings.
        :return: A tuple containing the x and y coordinates.
        """
        if isinstance(class_ds, Dataset):
            return class_ds[col_embs][:, 0], class_ds[col_embs][:, 1]
        else:
            xs = {class_label: class_ds[class_label][col_embs][:, 0] for class_label in class_ds}
            ys = {class_label: class_ds[class_label][col_embs][:, 1] for class_label in class_ds}
            return xs, ys
    

    @staticmethod
    def _zip_classes_to_values(class_ds: SplitDataset, values: list) -> dict:
        """
        This method zips the classes to the values, i.e. it creates a dictionary containing the classes 
        zipped to the values of a given list.
        
        :param class_ds: The dataset containing the classes.
        :param values: The values to zip.
        :return: A dictionary containing the classes zipped to the values.
        """
        subvalues: list[Any] = values[:len(class_ds)]
        return {class_label: subvalues[i] for i, class_label in enumerate(class_ds)}