"""
Tensor Flow examples.
"""

import os
import sys
import typing
import logging

from functools import lru_cache

import skimage
import tensorflow as tf

import matplotlib.pyplot as plt

# Printing logs to console.
# Reference: https://stackoverflow.com/questions/14058453
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class File:
    """
    File System file entity.
    """

    def __init__(self, path: str) -> None:
        """
        File constructor.
        """
        logger.debug("Constructing %s | sf_path=%s", self.__class__.__name__, path)
        if not isinstance(path, str):
            raise TypeError("Expecting str, got:", type(path))
        if not path:
            raise ValueError("Path is required.")
        if not os.path.exists(path):
            raise OSError("Path not found:", path)
        if not self.is_valid(path):
            raise OSError("Invalid path:", path)
        self.__path: str = path

    @classmethod
    def is_valid(cls, path: str) -> bool:
        """
        Public class method to validate if name is valid.
        """
        if not isinstance(path, str):
            raise TypeError("Expecting str, got:", type(path))
        if not path:
            raise ValueError("Name is required.")
        if not os.path.exists(path):
            raise OSError("Path not found:", path)
        return os.path.isfile(path)

    def __str__(self) -> str:
        """
        File string serializer.
        """
        return "<{}: '{}'>".format(self.__class__.__name__, self.path)

    @property
    def name(self) -> str:
        """
        File name getter.
        """
        return self.path.split("/")[-1]

    @property
    def path(self) -> str:
        """
        File path getter.
        """
        return self.__path


class Directory(File):
    """
    File System directory entity.
    """

    @classmethod
    def is_valid(cls, path: str) -> bool:
        """
        Public class method to validate if name is valid.
        """
        if not isinstance(path, str):
            raise TypeError("Expecting str, got:", type(path))
        if not path:
            raise ValueError("Name is required.")
        if not os.path.exists(path):
            raise OSError("Path not found:", path)
        return os.path.isdir(path)

    @property
    @lru_cache()
    def files(self) -> typing.Set[File]:
        """
        Public method to get files in directory.
        """
        logger.debug("Listing files | sf_path=%s", self.path)
        files = (
            os.path.join(self.path, file_name)
            for file_name in os.listdir(self.path)
        )
        return {
            File(file_path)
            for file_path in files
            if os.path.isfile(file_path)
        }

    @property
    @lru_cache()
    def directories(self) -> typing.Set['Directory']:
        """
        Public method to get sub directories in directory.
        """
        logger.debug("Listing directories | sf_path=%s", self.path)
        directories = (
            os.path.join(self.path, dir_name)
            for dir_name in os.listdir(self.path)
        )
        return {
            Directory(dir_path)
            for dir_path in directories
            if os.path.isdir(dir_path)
        }

    @property
    def total(self) -> int:
        """
        Getter of the amount of elements in dir.
        """
        return len(self.files) + len(self.directories)


class Image(File):
    """
    Image entity.
    """

    SUFFIX = ".ppm"

    @classmethod
    def is_valid(cls, path: str) -> bool:
        """
        Public class method to validate if name is valid.
        """
        if not isinstance(path, str):
            raise TypeError("Expecting str, got:", type(path))
        if not path:
            raise ValueError("Name is required.")
        return File.is_valid(path) and path.endswith(cls.SUFFIX)

    def to_array(self) -> list:
        """
        Array serializer.
        For example:
        >>> [
        ...     [65 42 46]
        ...     [64 42 46]
        ...     [64 41 45]
        ...     [...]
        ... ]
        """
        logger.debug("Exporting to Array | sf_path=%s", self.path)
        raise Exception(type(skimage.data.imread(self.path)))
        return skimage.data.imread(self.path)


class Label(Directory):
    """
    Data label entity.
    """

    @property
    @lru_cache()
    def images(self) -> typing.Set[Image]:
        """
        Public method to access data files.
        For example:
        >>> {
        ...     Image("data/00011/a.img"),
        ...     Image("data/00011/b.img"),
        ...     Image("data/00011/c.img"),
        ...     [...]
        ... }
        """
        logger.debug("Listing images | sf_path=%s", self.path)
        return {
            Image(file_.path)
            for file_ in self.files
            if Image.is_valid(file_.path)
        }


class Dataset(Directory):
    """
    Dataset entity.
    """

    LABEL = "label"
    IMAGE = "image"

    @property
    @lru_cache()
    def labels(self) -> typing.Set[Label]:
        """
        Data labels getter.
        For example:
        >>> {
        ...     Label("00011"),
        ...     Label("00021"),
        ...     Label("00013"),
        ...     [...]
        ... }
        """
        logger.debug("Listing labels | sf_path=%s", self.path)
        return {
            Label(directory.path)
            for directory in self.directories
        }

    @property
    @lru_cache()
    def frequencies(self) -> dict:
        """
        Public method to return the frequencies for each label.
        """
        logger.debug("Generating Frequencies | sf_path=%s", self.path)
        return {
            label.path: label.total
            for label in self.labels
        }

    @property
    @lru_cache()
    def histogram(self) -> list:
        """
        Public method to return the histogram.
        """
        logger.debug("Generating Histogram | sf_path=%s", self.path)
        return [
            label.name
            for label in self.labels
            for _ in label.images
        ]

    def to_json(self) -> typing.Generator:
        """
        Dataset serializer.
        For example:
        >>> [
        ...     [
        ...         [65 42 46]
        ...         [64 42 46]
        ...         [64 41 45]
        ...         [...]
        ...         [61 44 46]
        ...         [58 41 44]
        ...         [57 40 42]
        ...     ],
        ...     [...]
        ... ]
        """
        logger.debug("Exporting to JSON | sf_path=%s", self.path)
        return (
            {
                self.LABEL: label.name,
                self.IMAGE: image.to_array()
            }
            for label in self.labels
            for image in label.images
        )


class Diagram:
    """
    Chat diagram entity.
    """

    FONT_SIZE = 16
    TITLE = "My Little Diagram"
    LABELS = ("", "")

    ROWS = 1
    COLS = 1

    PATH = os.path.join("plot")
    NAME = ""

    def __init__(self) -> None:
        """
        Diagram constructor.
        """
        logger.debug("Constructing Diagram.")
        self.__figure: typing.Optional['matplotlib.figure.Figure'] = None
        self.__axis : typing.Optional['matplotlib.axes._subplots.AxesSubplot'] = None

    def __str__(self) -> str:
        """
        String serializer.
        """
        return "<{}>".format(self.__class__.__name__)

    @property
    def path(self) -> str:
        """
        Diagram path getter.
        """
        if not self.NAME:
            raise RuntimeError("Expecting diagram file name.")
        return os.path.join(self.PATH, self.NAME)

    def __get_subplot(self) -> typing.Tuple:
        """
        Private Subplot generator.
        """
        return plt.subplots(self.ROWS, self.COLS, constrained_layout=True)

    @property
    def figure(self) -> 'matplotlib.figure.Figure':
        """
        Maptlot figure getter.
        """
        if self.__figure is None:
            self.__figure, self.__axis = self.__get_subplot()
        return self.__figure

    @property
    def axis(self) -> 'matplotlib.axes._subplots.AxesSubplot':
        """
        Maptlot axis getter.
        """
        if self.__axis is None:
            self.__figure, self.__axis = self.__get_subplot()
        return self.__axis

    def draw(self, *args, **kwargs) -> None:
        """
        Public method to draw diagram.
        """
        logger.debug("Drawing %s", self)
        self.figure.suptitle(self.TITLE, fontsize=self.FONT_SIZE)
        self.figure.savefig(self.path)


class DatasetHistogram(Diagram):
    """
    Dataset Histogram chart.
    """

    NAME = "data_histogram.png"

    TITLE = "Data Histogram"
    LABELS = ("Labels", "Images")

    ROWS = 2
    COLS = 1

    TRAIN_SUBTITLE = "Training Labels"
    TEST_SUBTITLE = "Testing Labels"

    def __init__(self, test_set: Dataset, train_set: Dataset) -> None:
        """
        Diagram constructor.
        """
        logger.debug("Constructing Dataset Histogram.")
        if not isinstance(train_set, Dataset):
            raise TypeError("Expecting Dataset, got:", type(train_set))
        if not isinstance(test_set, Dataset):
            raise TypeError("Expecting Dataset, got:", type(test_set))
        self.__test_set = test_set
        self.__train_set = train_set
        Diagram.__init__(self)

    def draw(self) -> None:
        """
        Public method to draw histograms.
        """
        logger.debug("Drawing Histogram | sf_train=%s | sf_test=%s",
                     self.__train_set, self.__test_set)
        self.axis[0].hist(self.__train_set.histogram,
                          len(self.__train_set.frequencies))
        self.axis[0].set_title(self.TRAIN_SUBTITLE)
        self.axis[0].set_xlabel(self.LABELS[0])
        self.axis[0].set_ylabel(self.LABELS[1])
        self.axis[1].hist(self.__test_set.histogram,
                          len(self.__test_set.frequencies))
        self.axis[1].set_title(self.TEST_SUBTITLE)
        self.axis[1].set_xlabel(self.LABELS[0])
        self.axis[1].set_ylabel(self.LABELS[1])
        Diagram.draw(self)


class Main:
    """
    Main class.
    """

    TESTING_DATASET = os.path.join("data", "testing")
    TRAINING_DATASET = os.path.join("data", "training")

    @classmethod
    def run(cls) -> None:
        """
        Main handler.
        """
        main = cls()
        main.draw()

    def __init__(self) -> None:
        """
        Main constructor.
        """
        logger.debug("Initializing Main.")
        self.__train_set: Dataset = Dataset(self.TRAINING_DATASET)
        self.__test_set: Dataset = Dataset(self.TESTING_DATASET)

    def draw(self) -> None:
        """
        Public method to draw diagrams.
        """
        logger.debug("Drawing diagrams.")
        DatasetHistogram(self.__train_set, self.__test_set).draw()

if __name__ == "__main__":
    Main.run()
