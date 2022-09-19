import string
import tensorflow as tf
from typing import re
from tensorflow.keras.layers import TextVectorization

import TextPreprocessing.TextPreprocess


class Vectorization:
    def __init__(self, raw_train_ds, max_features, embedding_dim, sequence_length):
        self.raw_train_ds = raw_train_ds
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.vectorize_layer = None

        self.build_standard_vectorize_layer()

    def build_standard_vectorize_layer(self):
        # Now that we have our custom standardization, we can instantiate our text
        # vectorization layer. We are using this layer to normalize, split, and map
        # strings to integers, so we set our 'output_mode' to 'int'.
        # Note that we're using the default split function,
        # and the custom standardization defined above.
        # We also set an explicit maximum sequence length, since the CNNs later in our
        # model won't support ragged sequences.

        self.vectorize_layer = TextVectorization(
            standardize=TextPreprocessing.TextPreprocess.preprocess,
            max_tokens=self.max_features,
            output_mode="int",
            output_sequence_length=self.sequence_length,
        )

        # Now that the vocab layer has been created, call `adapt` on a text-only
        # dataset to create the vocabulary. You don't have to batch, but for very large
        # datasets this means you're not keeping spare copies of the dataset in memory.

        # Let's make a text-only dataset (no labels):
        text_ds = self.raw_train_ds.map(lambda x, y: x)
        # Let's call `adapt`:
        self.vectorize_layer.adapt(text_ds)

    def vectorize_text(self, text, label):
        text = tf.expand_dims(text, -1)
        return self.vectorize_layer(text), label

    def get_vectorize_layer(self):
        return self.vectorize_layer
