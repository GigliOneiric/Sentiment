import os
import shutil
import tensorflow as tf


class IMDB:

    def __init__(self):
        self.raw_train_ds = None
        self.raw_val_ds = None
        self.raw_test_ds = None
        self.dataset = None

        self.load_datasets()
        self.split_datasets()

    def load_datasets(self):
        """
        ## Load the data: IMDB movie review sentiment classification.
        """

        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

        self.dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                               untar=True, cache_dir='.',
                                               cache_subdir='')

        """
        ## Set paths to the directory's
        """

        dataset_dir = os.path.join(os.path.dirname(self.dataset), 'aclImdb')
        train_dir = os.path.join(dataset_dir, 'train')

        """
        ## Remove unused directory
        """

        remove_dir = os.path.join(train_dir, 'unsup')
        shutil.rmtree(remove_dir)

        """
        The utility `tf.keras.preprocessing.text_dataset_from_directory` is used to
        generate a labeled `tf.data.Dataset` object from a set of text files on disk filed
        into class-specific folders.

        Let's use it to generate the training, validation, and test datasets. The validation
        and training datasets are generated from two subsets of the `train` directory, with 20%
        of samples going to the validation dataset and 80% going to the training dataset.
        """

    def split_datasets(self):
        batch_size = 32
        self.raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
            "aclImdb/train",
            batch_size=batch_size,
            validation_split=0.2,
            subset="training",
            seed=1337,
        )
        self.raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
            "aclImdb/train",
            batch_size=batch_size,
            validation_split=0.2,
            subset="validation",
            seed=1337,
        )
        self.raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
            "aclImdb/test", batch_size=batch_size
        )

    def get_train_set(self) -> tf.data.Dataset:
        return self.raw_train_ds

    def get_test_set(self) -> tf.data.Dataset:
        return self.raw_test_ds

    def get_val_set(self) -> tf.data.Dataset:
        return self.raw_val_ds