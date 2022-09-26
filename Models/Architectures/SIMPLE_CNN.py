import tensorflow as tf
from tensorflow.keras import layers

from Models.Architectures.Layers.Embeddings import Embeddings
from Models.Architectures.Layers.Output.Prediction import create_prediction_layer


class SIMPLE_CNN:

    def __init__(self, max_features, embedding_type, embedding_dim, sequence_length, vectorization_layer,
                 hidden_layers, filters, kernel_size, dense_units, dropout,
                 raw_test_ds, test_ds, train_ds, val_ds):
        self.raw_test_ds = raw_test_ds
        self.test_ds = test_ds
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.max_features = max_features
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length

        self.vectorization_layer = vectorization_layer

        self.hidden_layers: int = hidden_layers
        self.filters: int = filters
        self.kernel_size: int = kernel_size
        self.dense_units: int = dense_units
        self.dropout: float = dropout

    def create_cnn(self):
        # A integer input for vocab indices.
        inputs = tf.keras.Input(shape=(None,), dtype="int64")

        # Next, we add a layer to map those vocab indices into a space of dimensionality
        # 'embedding_dim'.
        x = Embeddings.choose_embeddings(self.max_features, self.embedding_dim, self.sequence_length,
                                         self.vectorization_layer, inputs, self.embedding_type)

        x = layers.Dropout(self.dropout)(x)

        # Conv1D + global max pooling
        for i in range(self.hidden_layers):
            x = layers.Conv1D(self.filters, self.kernel_size, padding="valid", activation="relu", strides=3)(x)

        x = layers.GlobalMaxPooling1D()(x)

        predictions = create_prediction_layer(x, self.dense_units, self.dropout)

        model = tf.keras.Model(inputs, predictions)

        # Compile the model with binary crossentropy loss and an adam optimizer.
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model
