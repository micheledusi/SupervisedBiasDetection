# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module contains the class "EmbeddingPlotter" which is used to plot the embeddings.

from datasets import Dataset
import matplotlib.pyplot as plt


def emb2plot(embeddings: Dataset) -> Dataset:
    """
    Converts a dataset with reduced embeddings into a dataset with the right information to be plotted.
    It requires:
    - an 'embedding' column, to be split into 'x' and 'y'.
    - an 

    :param embeddings: The dataset with embeddings information to be plotted.
    :return: A dataset that can be plotted by the ScatterPlotter class.
    """
    kvals: dict[str, int] = {}
    def extract_info_fn(sample):
        coords = sample["embedding"]
        assert len(coords) == 2, "Cannot draw a bidimensional plot with embedding of size = {}".format(len(coords))
        sample[ScatterPlotter.DEFAULT_COLUMN_X] = coords[0]
        sample[ScatterPlotter.DEFAULT_COLUMN_Y] = coords[1]
        # Label
        if ScatterPlotter.DEFAULT_COLUMN_LABEL not in sample:
            sample[ScatterPlotter.DEFAULT_COLUMN_LABEL] = sample['word']
        # Color
        if ScatterPlotter.DEFAULT_COLUMN_COLOR not in sample:
            # Mapping each key to a unique integer
            if sample['value'] not in kvals:
                kvals[sample['value']] = len(kvals)
            sample[ScatterPlotter.DEFAULT_COLUMN_COLOR] = kvals[sample['value']]
        return sample

    return embeddings.map(extract_info_fn, remove_columns=['embedding', 'word', 'value'], num_proc=1)


class ScatterPlotter:

    DEFAULT_COLUMN_X = "x"
    DEFAULT_COLUMN_Y = "y"
    DEFAULT_COLUMN_LABEL = "label"
    DEFAULT_COLUMN_COLOR = "color"

    def __init__(self, data: Dataset, title: str = None) -> None:
        """
        The initializer for the ScatterPlotter class.

        :param data: The dataset containing the embeddings.
        :param title: The title of the plot.
        """
        self._title = title
        self._x = data[self.DEFAULT_COLUMN_X]
        self._y = data[self.DEFAULT_COLUMN_Y]
        self._labels = data[self.DEFAULT_COLUMN_LABEL]
        self._colors = data[self.DEFAULT_COLUMN_COLOR]

    def show(self) -> None:
        """
        This method shows the plot.
        """
        plt.scatter(self._x, self._y, c=self._colors)
        for i, txt in enumerate(self._labels):
            plt.annotate(txt, (self._x[i], self._y[i]))
        plt.title(self._title)
        plt.show()