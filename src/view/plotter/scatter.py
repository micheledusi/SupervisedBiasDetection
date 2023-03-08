# - - - - - - - - - - - - - - - #
#   Supervised Bias Detection   #
#								#
#   Author:  Michele Dusi		#
#   Date:	2023				#
# - - - - - - - - - - - - - - - #

# This module contains the class "EmbeddingPlotter" which is used to plot the embeddings.

from datasets import Dataset
import matplotlib.pyplot as plt

DEFAULT_COLUMN_X = "x"
DEFAULT_COLUMN_Y = "y"
DEFAULT_COLUMN_LABEL = "word"
DEFAULT_COLUMN_COLOR = "value"


class ScatterPlotter:
    """
    This class is used to plot the embeddings.
    The only requirement is that the dataset contains the columns "x" and "y".
    """
    def __init__(self, data: Dataset, title: str = None, 
                 x_col: str = DEFAULT_COLUMN_X, y_col: str = DEFAULT_COLUMN_Y, 
                 label_col: str = DEFAULT_COLUMN_LABEL, color_col: str = DEFAULT_COLUMN_COLOR) -> None:
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