import os
from pathlib import Path

import numpy as np

from Data.Dataset import Dataset
from Models.Architectures.Layers.Input import Vectorization
from Models.Architectures.SIMPLE_GRU import SIMPLE_GRU
from Models.Architectures.SIMPLE_BiGRU import SIMPLE_BiGRU
from Models.Architectures.SIMPLE_CNN import SIMPLE_CNN
from Models.Training import Training
from Models.Export import Export

"""
## Load the dataset
"""



dataset_path = os.path.join(os.path.join(os.path.join(Path(os.getcwd()), 'Data'), 'Datasets'))
dataset_dir = os.path.join(dataset_path, 'aclImdb')
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

data = Dataset(dataset_path, dataset_dir, train_dir, test_dir)

raw_train_ds = data.get_train_set()
raw_val_ds = data.get_val_set()
raw_test_ds = data.get_test_set()

"""
## Prepare the data
"""

max_features = 10000
embedding_type = "glove"
embedding_dim = 300
sequence_length = 500

vectorization_layer = Vectorization.Vectorization(raw_train_ds, max_features, embedding_dim, sequence_length)

# Vectorize the data.
train_ds = raw_train_ds.map(vectorization_layer.vectorize_text)
val_ds = raw_val_ds.map(vectorization_layer.vectorize_text)
test_ds = raw_test_ds.map(vectorization_layer.vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)

"""
## Build a model
"""

hidden_layers = 1
rec_units = 128
dense_units = 128
dropout = 0.5

SIMPLE_BiGRU = SIMPLE_BiGRU(max_features, embedding_type, embedding_dim, sequence_length,
                            vectorization_layer.get_vectorize_layer(),
                            rec_units, dense_units, dropout,
                            raw_test_ds, test_ds, train_ds, val_ds)

model = SIMPLE_BiGRU.create_bigru()

"""
## Train the model
"""

epochs = 100
batch_size = 512

model = Training.train_model(model, epochs,
                             train_ds, val_ds, test_ds, True, batch_size)

"""
## Build the final model
"""

end_to_end_model = Export.build_end_to_end_model(model, vectorization_layer)

"""
## Play with the final model
"""

sample_text = 'Tesla is nice as shit'

sentiment = end_to_end_model.predict(np.array([sample_text]))

# If the prediction is >= 0.0, it is positive else it is negative.
print(sentiment)
