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


class Config:
    """
    App configuration.
    """
    TESTING_DATASET = os.path.join("data", "testing")
    TRAINING_DATASET = os.path.join("data", "training")


train: Dataset = Dataset(Config.TRAINING_DATASET)
test: Dataset = Dataset(Config.TESTING_DATASET)

fig, axis = plt.subplots(2, 1, constrained_layout=True)

axis[0].hist(train.histogram, 20)
axis[0].set_title('Training Dataset')
axis[0].set_xlabel('Labels')
axis[0].set_ylabel('Images')

axis[1].hist(test.histogram, 20)
axis[1].set_title('Testing Dataset')
axis[1].set_xlabel('Labels')
axis[1].set_ylabel('Images')

fig.suptitle('Data Histogram', fontsize=16)
fig.savefig('data_histogram.png')
