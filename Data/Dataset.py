import os

import tensorflow as tf

import Utils.files_preprocessing
from Data.FinancialPhrasebank import FinancialPhrasebank
from Data.IMDB import IMDB


class Dataset:

    def __init__(self, dataset_path, dataset_dir, train_dir, test_dir):
        self.dataset_path = dataset_path
        self.dataset_file = None
        self.dataset = None
        self.raw_train_ds = None
        self.raw_val_ds = None
        self.raw_test_ds = None
        self.dataset_dir = dataset_dir
        self.train_dir = train_dir
        self.train_pos_dir = os.path.join(train_dir, 'pos')
        self.train_neg_dir = os.path.join(train_dir, 'neg')
        self.test_dir = test_dir
        self.test_pos_dir = os.path.join(test_dir, 'pos')
        self.test_neg_dir = os.path.join(test_dir, 'neg')

        self.check_path()
        self.load_datasets()

        Utils.files_preprocessing.preprocess_all_folders()

        self.split_datasets()

    def check_path(self):
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
            os.makedirs(self.train_pos_dir)
            os.makedirs(self.train_neg_dir)

        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
            os.makedirs(self.test_pos_dir)
            os.makedirs(self.test_neg_dir)

    def load_datasets(self):
        #IMDB(self.dataset_path, self.dataset_dir, self.train_dir, self.test_dir)
        FinancialPhrasebank(self.dataset_path, self.dataset_dir, self.train_dir, self.test_dir,
                            self.train_pos_dir, self.train_neg_dir,
                            self.test_pos_dir, self.test_neg_dir)




    """
    ## The utility `tf.keras.preprocessing.text_dataset_from_directory` is used to
    ## generate a labeled `tf.data.Dataset` object from a set of text files on disk filed
    ## into class-specific folders.

    ## Let's use it to generate the training, validation, and test datasets. The validation
    ## and training datasets are generated from two subsets of the `train` directory, with 20%
    ## of samples going to the validation dataset and 80% going to the training dataset.
    """

    def split_datasets(self):
        batch_size = 32
        self.raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
            self.train_dir,
            batch_size=batch_size,
            validation_split=0.2,
            subset="training",
            seed=1337,
        )
        self.raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
            self.train_dir,
            batch_size=batch_size,
            validation_split=0.2,
            subset="validation",
            seed=1337,
        )
        self.raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
            self.test_dir, batch_size=batch_size
        )

    def get_train_set(self) -> tf.data.Dataset:
        return self.raw_train_ds

    def get_test_set(self) -> tf.data.Dataset:
        return self.raw_test_ds

    def get_val_set(self) -> tf.data.Dataset:
        return self.raw_val_ds
