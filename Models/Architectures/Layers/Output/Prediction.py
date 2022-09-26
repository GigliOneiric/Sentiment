import tensorflow as tf
from tensorflow.keras import layers


def create_prediction_layer(x, dense_units, dropout):
    # After the RNN has converted the sequence to a single vector the two layers.Dense do some final processing,
    # and convert from this vector representation to a single logit as the classification output.
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    # We project onto a single unit output layer, and squash it with a sigmoid:
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    return predictions
