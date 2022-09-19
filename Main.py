import tensorflow as tf
import string
from Data.IMDB import IMDB
from Models.Vectorization import Vectorization
from Models.Architectures import RNN

"""
## Load the dataset
"""

imdb_data = IMDB()

raw_train_ds = imdb_data.get_train_set()
raw_val_ds = imdb_data.get_val_set()
raw_test_ds = imdb_data.get_test_set()

"""
## Prepare the data
"""

max_features = 20000
embedding_dim = 128
sequence_length = 500

Vectorization = Vectorization.Vectorization(raw_train_ds, max_features, embedding_dim, sequence_length)
vectorize_text = Vectorization.vectorize_text


# Vectorize the data.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)

"""
## Build a model

"""
RNN = RNN.RNN(max_features, embedding_dim, Vectorization, raw_test_ds, test_ds, train_ds, val_ds)
RNN.create_rnn()
