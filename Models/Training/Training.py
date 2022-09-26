import numpy as np
import tensorflow as tf


def train_model(model, Vectorization, raw_test_ds, train_ds, val_ds, test_ds):
    """
     ## Train the model
    """

    epochs = 3

    # Fit the model using the train and test datasets.
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    """
    ## Evaluate the model on the test set
    """

    model.evaluate(test_ds)

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

    # Test it with `raw_test_ds`, which yields raw strings
    end_to_end_model.evaluate(raw_test_ds)

    sample_text = ('The movie was cool. The animation and the graphics '
                   'were out of this world. I would recommend this movie.')
    sentiment = end_to_end_model.predict(np.array([sample_text]))

    # If the prediction is >= 0.0, it is positive else it is negative.
    print(sentiment)
