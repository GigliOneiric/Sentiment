import os

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from pathlib import Path


def glove(max_features, embedding_dim, sequence_length, vectorization_layer: TextVectorization, inputs):
    glove_dir = download_glove()
    embedding_matrix = load_glove_vectors(glove_dir, vectorization_layer, max_features, embedding_dim)

    x = layers.Embedding(input_dim=max_features,
                         output_dim=embedding_dim,
                         input_length=sequence_length,
                         trainable=False,
                         weights=[embedding_matrix])(inputs)

    return x


def download_glove():
    url = "https://nlp.stanford.edu/data/glove.840B.300d.zip"

    glove_path = os.path.join(os.path.join(os.path.join(Path(os.getcwd()), 'Data'), 'Datasets'))

    tf.keras.utils.get_file("glove.840B.300d", url,
                            untar=True, cache_dir=glove_path,
                            cache_subdir='')


    """
    ## Set paths to the directory's
    """

    glove_dir = os.path.join(glove_path, 'glove.840B.300d')
    glove_file = os.path.join(glove_dir, 'glove.840B.300d.txt')

    return glove_file


def load_glove_vectors(glove_file, vectorization_layer, max_features, embedding_dim):
    voc = vectorization_layer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    """
    The archive contains text-encoded vectors of various sizes: 50-dimensional,
    100-dimensional, 200-dimensional, 300-dimensional. We'll use the 300D ones.
    Let's make a dict mapping words (strings) to their NumPy vector representation:
    """

    embeddings_index = {}
    with open(glove_file, encoding="utf8") as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            weights = np.asarray([float(val) for val in values[1:]])
            embeddings_index[word] = weights

    """
     Now, let's prepare a corresponding embedding matrix that we can use in a Keras
     `Embedding` layer. It's a simple NumPy matrix where entry at index `i` is the pre-trained
     vector for the word of index `i` in our `vectorizer`'s vocabulary.
     """

    embedding_dim = embedding_dim
    if max_features is not None:
        vocab_len = max_features
    else:
        vocab_len = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_len, embedding_dim))
    oov_count = 0
    oov_words = []
    for word, idx in word_index.items():
        if idx < vocab_len:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
            else:
                oov_count += 1
                oov_words.append(word)

    return embedding_matrix
