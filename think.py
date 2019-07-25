"""
Tensor Flow examples.
"""

import os
import typing
import random
import logging

from functools import lru_cache
from slugify import slugify

import numpy as np

# import cv2
import skimage
from skimage.color import rgb2gray
import tensorflow as tf

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


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


class ImageFile(File):
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

    @lru_cache()
    def to_array(self, size: int = None, gray: bool = False) -> np.array:
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
        logger.debug("Exporting to Array | sf_path=%s | sf_size=%s", self.path, size)
        array: np.ndarray = skimage.data.imread(self.path)
        if size is not None:
            if not isinstance(size, int):
                raise TypeError("Expecting int, got:", type(size))
            if size < 1:
                raise ValueError("Invalid size.")
            array = skimage.transform.resize(array, (size, size))
        if gray:
            array = rgb2gray(array)
        return array


class LabelDirectory(Directory):
    """
    Data label entity.
    """

    @property
    @lru_cache()
    def images(self) -> typing.Set[ImageFile]:
        """
        Public method to access data files.
        For example:
        >>> {
        ...     ImageFile("data/00011/a.img"),
        ...     ImageFile("data/00011/b.img"),
        ...     ImageFile("data/00011/c.img"),
        ...     [...]
        ... }
        """
        logger.debug("Listing images | sf_path=%s", self.path)
        return {
            ImageFile(file_.path)
            for file_ in self.files
            if ImageFile.is_valid(file_.path)
        }


class Dataset(Directory):
    """
    Dataset entity.
    """

    LABEL = "label"
    IMAGE = "image"

    @property
    @lru_cache()
    def labels(self) -> typing.Set[LabelDirectory]:
        """
        Data labels getter.
        For example:
        >>> {
        ...     LabelDirectory("00011"),
        ...     LabelDirectory("00021"),
        ...     LabelDirectory("00013"),
        ...     [...]
        ... }
        """
        logger.debug("Listing labels | sf_path=%s", self.path)
        return {
            LabelDirectory(directory.path)
            for directory in self.directories
        }

    @property
    @lru_cache()
    def images(self) -> typing.Set[ImageFile]:
        """
        Data images getter.
        For example:
        >>> {
        ...     ImageFile("00011"),
        ...     ImageFile("00012"),
        ...     ImageFile("00013"),
        ...     [...]
        ... }
        """
        logger.debug("Listing images | sf_path=%s", self.path)
        return {
            image
            for label in self.labels
            for image in label.images
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

    @lru_cache()
    def to_array(self,
                 size: int = None,
                 gray: bool = False) -> typing.List[typing.List]:
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
        return [
            [image.to_array(size=size, gray=gray), label.name]
            for label in self.labels
            for image in label.images
        ]


class Diagram:
    """
    Chat diagram entity.
    """

    FONT_SIZE = 12
    TITLE = "My Little Diagram"
    LABELS = ("", "")

    ROWS = 1
    COLS = 1

    PATH = os.path.join("plot")
    FORMAT = ".png"

    def __init__(self) -> None:
        """
        Diagram constructor.
        """
        logger.debug("Constructing Diagram.")
        self.__figure: typing.Optional[plt.figure] = None
        self.__axis: typing.Optional[plt.axes] = None

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
        return os.path.join(self.PATH, slugify(self.TITLE) + self.FORMAT)

    def __get_subplot(self) -> typing.Tuple:
        """
        Private Subplot generator.
        """
        return plt.subplots(self.ROWS, self.COLS, constrained_layout=True)

    @property
    def figure(self) -> plt.figure:
        """
        Maptlot figure getter.
        """
        if self.__figure is None:
            self.__figure, self.__axis = self.__get_subplot()
        return self.__figure

    @property
    def axis(self) -> plt.axes:
        """
        Maptlot axis getter.
        """
        if self.__axis is None:
            self.__figure, self.__axis = self.__get_subplot()
        return self.__axis

    def draw(self) -> None:
        """
        Public method to draw diagram.
        """
        logger.debug("Drawing %s", self)
        self.figure.suptitle(self.TITLE, fontsize=self.FONT_SIZE)
        self.figure.savefig(self.path)


class DatasetDiagram(Diagram):
    """
    Dataset chart.
    """

    def __init__(self, test_set: Dataset, train_set: Dataset) -> None:
        """
        Diagram constructor.
        """
        logger.debug("Constructing Dataset Diagram.")
        if not isinstance(train_set, Dataset):
            raise TypeError("Expecting Dataset, got:", type(train_set))
        if not isinstance(test_set, Dataset):
            raise TypeError("Expecting Dataset, got:", type(test_set))
        self.__test_set = test_set
        self.__train_set = train_set
        Diagram.__init__(self)

    @property
    def train_set(self):
        """
        Train set getter.
        """
        return self.__train_set

    @property
    def test_set(self):
        """
        Test set getter.
        """
        return self.__test_set


class DatasetHistogramDiagram(DatasetDiagram):
    """
    Dataset Histogram chart.
    """

    TITLE = "Data Histogram"
    LABELS = ("Labels", "Images")

    ROWS = 2
    COLS = 1

    TRAIN_SUBTITLE = "Training Labels"
    TEST_SUBTITLE = "Testing Labels"

    def draw(self) -> None:
        """
        Public method to draw histograms.
        """
        logger.debug("Drawing Histogram | sf_train=%s | sf_test=%s", self.train_set, self.test_set)
        self.axis[0].hist(self.train_set.histogram, len(self.train_set.frequencies))
        self.axis[0].set_title(self.TRAIN_SUBTITLE)
        self.axis[0].set_xlabel(self.LABELS[0])
        self.axis[0].set_ylabel(self.LABELS[1])
        self.axis[1].hist(self.test_set.histogram, len(self.test_set.frequencies))
        self.axis[1].set_title(self.TEST_SUBTITLE)
        self.axis[1].set_xlabel(self.LABELS[0])
        self.axis[1].set_ylabel(self.LABELS[1])
        Diagram.draw(self)


class DatasetSampleDiagram(DatasetDiagram):
    """
    Diagram that shows random data.
    """

    TITLE = "Random Sample"

    ROWS = 3
    COLS = 4

    def draw(self) -> None:
        """
        Public method to draw histograms.
        """
        logger.debug("Drawing Sample | sf_train=%s | sf_test=%s", self.train_set, self.test_set)
        sample = random.sample(self.train_set.images, self.COLS * self.ROWS)
        for i in range(self.ROWS):
            for j in range(self.COLS):
                self.axis[i][j].imshow(skimage.data.imread(sample.pop().path))
                self.axis[i][j].axis('off')
        Diagram.draw(self)


class SamplePredictionDiagram(Diagram):
    """
    Sample prediction diagram.
    """

    TITLE = "Sample Predictions"

    ROWS = 5
    COLS = 3

    WSPACE = 0.5

    def __init__(self, x: list, y: list, y_hat: np.ndarray) -> None:
        """
        Diagram constructor.
        """
        logger.debug("Constructing Sample Dataset Diagram.")
        if not isinstance(x, list):
            raise TypeError("Expecting list, got:", type(x))
        if not isinstance(y, list):
            raise TypeError("Expecting list, got:", type(y))
        if not isinstance(y_hat, list):
            raise TypeError("Expecting array, got:", type(y_hat))
        if not x:
            raise ValueError("Empty sample training X set.")
        if not y:
            raise ValueError("Empty sample training Y set.")
        if not y_hat:
            raise ValueError("Empty sample predicted set.")
        if len(x) != len(y_hat):
            raise ValueError("Training and predicted samples' size not equal.")
        if len(y) != len(y_hat):
            raise ValueError("Training and predicted samples' size not equal.")
        if self.COLS * self.ROWS > len(x):
            raise RuntimeError("Sample training set is too small. Try:", self.COLS * self.ROWS)
        self.__x = x
        self.__y = y
        self.__y_hat = y_hat
        Diagram.__init__(self)

    def draw(self) -> None:
        """
        Public method to draw histograms.
        """
        logger.debug("Drawing Sample | sf_train=%s | sf_predicted=%s", self.__y, self.__y_hat)
        # fig = plt.figure(figsize=(10, 10))
        k = 0
        for i in range(self.ROWS):
            for j in range(self.COLS):
                logger.debug("Drawing Sample | sf_image=%s", k)
                truth = self.__y[k]
                prediction = self.__y_hat[k]
                color = 'green' if truth == prediction else 'red'
                self.axis[i][j].axis('off')
                text = "Y: {}\nȲ̂: {}".format(truth, prediction)
                self.axis[i][j].text(40, 10, text, fontsize=self.FONT_SIZE, color=color)
                self.axis[i][j].imshow(self.__x[k])
                k += 1
        Diagram.draw(self)


class PredictionModel:
    """
    Prediction Model class.
    """

    def __init__(self, sample_size: int, epochs: int, path: str,
                 learning_rate: float, scale: int, seed: int) -> None:
        """
        Main constructor.
        """
        logger.debug("Initializing Prediction Model.")
        if not isinstance(path, str):
            raise TypeError("Expecting str, got:", type(path))
        if not isinstance(learning_rate, float):
            raise TypeError("Expecting float, got:", type(learning_rate))
        if not isinstance(seed, int):
            raise TypeError("Expecting int, got:", type(seed))
        if not isinstance(scale, int):
            raise TypeError("Expecting int, got:", type(scale))
        if not isinstance(epochs, int):
            raise TypeError("Expecting int, got:", type(epochs))
        if not isinstance(sample_size, int):
            raise TypeError("Expecting sample_size, got:", type(sample_size))
        self.__path = path
        self.__learning_rate = learning_rate
        self.__sample_size = sample_size
        self.__epochs = epochs
        self.__scale = scale
        self.__seed = seed

    def __str__(self) -> str:
        """
        String serializer.
        """
        return "<PredictionModel: {}".format(self.__learning_rate)

    def learn(self, train_set: Dataset, test_set: Dataset) -> typing.Tuple:
        """
        Pubic method to learn.
        """
        logger.info("Learning. | sf_train=%s | sf_test=%s", train_set, test_set)
        if not isinstance(train_set, Dataset):
            raise TypeError("Expecting Dataset, got:", type(train_set))
        if not isinstance(test_set, Dataset):
            raise TypeError("Expecting Dataset, got:", type(test_set))
        # Defining placeholders.
        x = tf.placeholder(dtype=tf.float32, shape=[None, self.__scale, self.__scale])
        y = tf.placeholder(dtype=tf.int32, shape=[None])
        images_flat = tf.contrib.layers.flatten(x)
        # Defining activation function.
        logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
        logger.debug("Training | sf_logits=%s", logits)
        # Defining loss function.
        tmp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(tmp)
        logger.debug("Training | sf_loss=%s", loss)
        # Defining an optimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate=self.__learning_rate).minimize(loss)
        logger.debug("Training | sf_optimizer=%s", optimizer)
        # Converting logits to label indexes.
        correct_pred = tf.argmax(logits, 1)
        logger.debug("Training | sf_argmax=%s", correct_pred)
        # Defining accuracy metric.
        accuracy_metric = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        logger.debug("Training | sf_accuracy_metric=%s", accuracy_metric)
        # Setting random rseed.
        tf.compat.v1.set_random_seed(self.__seed)
        logger.debug("Training | sf_random_seed=%s", self.__seed)
        # Generating train and test datasets.
        train = train_set.to_array(size=self.__scale, gray=True)
        x_train = [x[0] for x in train]
        y_train = [x[1] for x in train]
        test = test_set.to_array(size=self.__scale, gray=True)
        x_test = [x[0] for x in test]
        y_test = [x[1] for x in test]
        logger.debug("Training | sf_training=%s | sf_testing=%s", len(train), len(test))
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        logger.debug("Training | sf_saver=%s", saver)
        # Training the model.
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.__epochs):
                logger.debug('Training | sf_epoch=%s', epoch)
                feed_dict = {x: x_train, y: y_train}
                sess.run([optimizer, accuracy_metric], feed_dict=feed_dict)
                if not epoch % 10:
                    logger.info("Training | sf_epoch=%s | sf_loss=%s", epoch, loss)
            # Calculating accuracy
            y_predicted = sess.run([correct_pred], feed_dict={x: x_test})[0]
            match_count = sum(
                int(y == y_hat)
                for y, y_hat in zip(y_test, y_predicted)
            )
            accuracy = match_count / len(y_test)
            logger.info("Trained. | sf_accuracy=%s", accuracy)
            # Obtaining a sample to draw a chart.
            sample_indexes = random.sample(range(len(x_test)), self.__sample_size)
            sample_x = [x_test[i] for i in sample_indexes]
            sample_y = [y_test[i] for i in sample_indexes]
            sample_y_hat = sess.run([correct_pred], feed_dict={x: sample_x})[0]
            logger.debug("Trained. | sf_expected=%s | sf_predicted=%s", sample_y, sample_y_hat)
            # Saving model to pkl file.
            saver.save(sess, self.__path)
        return accuracy, sample_x, sample_y, sample_y


class Config:
    """
    Default config.
    """
    TRAINING_SAMPLE_SIZE = 15
    LOG_LEVEL = logging.INFO
    TESTING_DATASET_PATH = os.path.join("data", "testing")
    TRAINING_DATASET_PATH = os.path.join("data", "training")
    IMAGES_SCALE = 28
    LEARNING_RATE = 0.001
    RANDOM_SEED = 1234
    EPOCHS = 100
    MODEL_PATH = os.path.join("model.cpkl")


class Main:
    """
    Main class.
    """

    @classmethod
    def run(cls, **kwargs) -> None:
        """
        Main handler.
        """
        main = cls(**kwargs)
        main.learn()
        main.draw()

    def __init__(self,
                 learning_rate: float = Config.LEARNING_RATE,
                 images_scale: int = Config.IMAGES_SCALE,
                 log_level: int = Config.LOG_LEVEL,
                 epochs: int = Config.EPOCHS,
                 model_path: str = Config.MODEL_PATH,
                 random_seed: int = Config.RANDOM_SEED,
                 training_sample_size: int = Config.TRAINING_SAMPLE_SIZE,
                 train_set_path: str = Config.TRAINING_DATASET_PATH,
                 test_set_path: str = Config.TESTING_DATASET_PATH) -> None:
        """
        Main constructor.
        """
        logger.debug("Initializing Main.")
        logger.setLevel(log_level)
        self.__train_set: Dataset = Dataset(train_set_path)
        self.__test_set: Dataset = Dataset(test_set_path)
        self.__accuracy: float = 0
        self.__sample_x: list = []
        self.__sample_y: list = []
        self.__sample_y_hat: list = []
        self.__model: PredictionModel = PredictionModel(learning_rate=learning_rate,
                                                        sample_size=training_sample_size,
                                                        epochs=epochs,
                                                        path=model_path,
                                                        seed=random_seed,
                                                        scale=images_scale)

    def draw(self) -> None:
        """
        Public method to draw diagrams.
        """
        logger.debug("Drawing diagrams.")
        SamplePredictionDiagram(self.__sample_x,
                                self.__sample_y,
                                self.__sample_y_hat).draw()
        DatasetSampleDiagram(self.__train_set, self.__test_set).draw()
        DatasetHistogramDiagram(self.__train_set, self.__test_set).draw()

    def learn(self) -> None:
        """
        Public method to learn from the training dataset.
        """
        logger.debug("Training the Prediction Model.")
        self.__accuracy, self.__sample_x,\
            self.__sample_y, self.__sample_y_hat = self.__model.learn(self.__train_set,
                                                                      self.__test_set)


if __name__ == "__main__":

    # Printing logs to console.
    # Reference: https://stackoverflow.com/questions/14058453
    # handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)

    # Running Main handler.
    Main.run()
