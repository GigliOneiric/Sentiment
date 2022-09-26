import numpy as np

from Data.IMDB import IMDB
from Models.Architectures.Layers.Input import Vectorization
from Models.Architectures.SIMPLE_GRU import SIMPLE_GRU
from Models.Architectures.SIMPLE_CNN import SIMPLE_CNN
from Models.Training import Training
from Models.Export import Export

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

max_features = 10000
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

SIMPLE_GRU = SIMPLE_GRU(max_features, embedding_dim, sequence_length,
                        hidden_layers, rec_units, dense_units, dropout,
                        raw_test_ds, test_ds, train_ds, val_ds)

model = SIMPLE_GRU.create_gru()

"""
## Train the model
"""

epochs = 100

model = Training.train_model(model, epochs,
                             train_ds, val_ds, test_ds, True)

"""
## Build the final model
"""

end_to_end_model = Export.build_end_to_end_model(model, vectorization_layer)

"""
## Play with the final model
"""

sample_text = ('Tesla is so cool'
               'The stock is super')
sentiment = end_to_end_model.predict(np.array([sample_text]))

# If the prediction is >= 0.0, it is positive else it is negative.
print(sentiment)
