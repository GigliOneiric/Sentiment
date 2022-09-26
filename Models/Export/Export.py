import tensorflow as tf


def build_end_to_end_model(model, Vectorization):
    """
    ## Make an end-to-end model

    If you want to obtain a model capable of processing raw strings, you can simply
    create a new model (using the weights we just trained):
    """

    # A string input
    inputs = tf.keras.Input(shape=(1,), dtype="string")
    # Turn strings into vocab indices
    vectorize_layer = Vectorization.get_vectorize_layer()
    indices = vectorize_layer(inputs)
    # Turn vocab indices into predictions
    outputs = model(indices)

    # Our end to end model
    end_to_end_model = tf.keras.Model(inputs, outputs)
    end_to_end_model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return end_to_end_model
