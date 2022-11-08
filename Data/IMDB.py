import os
import shutil

import tensorflow as tf


class IMDB:

    def __init__(self, dataset_path, dataset_dir, train_dir, test_dir):
        self.dataset_path = dataset_path
        self.dataset_file = None
        self.dataset = None
        self.dataset_dir = dataset_dir
        self.train_dir = train_dir
        self.test_dir = test_dir

        self.load_datasets()

    def load_datasets(self):

        """
        ## Set paths to the directory's
        """

        self.dataset_file = os.path.join(self.dataset_path, 'aclImdb_v1.tar.gz')

        """
        ## Load the data: IMDB movie review sentiment classification.
        """

        if not os.path.exists(self.dataset_file):

            url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

            self.dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                                   untar=True, cache_dir=self.dataset_path,
                                                   cache_subdir='')

            """
            ## Remove unused directory
            """

            remove_dir = os.path.join(self.train_dir, 'unsup')

            if os.path.isdir(remove_dir):
                shutil.rmtree(remove_dir)


