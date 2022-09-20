import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import SimpleRNN, Dense, Activation


class SIMPLE_RNN:

    def __init__(self, max_features, embedding_dim, Vectorization, hidden_layers, raw_test_ds, test_ds, train_ds, val_ds):
        self.raw_test_ds = raw_test_ds
        self.test_ds = test_ds
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.max_features = max_features
        self.embedding_dim = embedding_dim

        self.Vectorization = Vectorization

        self.hidden_layers = hidden_layers

    def create_rnn(self):
        # A integer input for vocab indices.
        inputs = tf.keras.Input(shape=(None,), dtype="int64")

        # Next, we add a layer to map those vocab indices into a space of dimensionality
        # 'embedding_dim'.
        x = layers.Embedding(self.max_features, self.embedding_dim)(inputs)
        x = layers.Dropout(0.5)(x)

        # Next, we add the RNN
        if self.hidden_layers > 1:
            for i in range(1, self.hidden_layers):
                x = SimpleRNN(units=60, return_sequences=True)(x)

        x = SimpleRNN(units=60)(x)

        # After the RNN has converted the sequence to a single vector the two layers.Dense do some final processing,
        # and convert from this vector representation to a single logit as the classification output.
        x = Dense(units=60)(x)
        x = Activation("relu")(x)

        # We project onto a single unit output layer, and squash it with a sigmoid:
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

        model = tf.keras.Model(inputs, predictions)

        # Compile the model with binary crossentropy loss and an adam optimizer.
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        """
        ## Train the model
        """

        epochs = 3

        # Fit the model using the train and test datasets.
        model.fit(self.train_ds, validation_data=self.val_ds, epochs=epochs)

        """
        ## Evaluate the model on the test set
        """

        model.evaluate(self.test_ds)

        """
        ## Make an end-to-end model
        
        If you want to obtain a model capable of processing raw strings, you can simply
        create a new model (using the weights we just trained):
        """

        # A string input
        inputs = tf.keras.Input(shape=(1,), dtype="string")
        # Turn strings into vocab indices
        vectorize_layer = self.Vectorization.get_vectorize_layer()
        indices = vectorize_layer(inputs)
        # Turn vocab indices into predictions
        outputs = model(indices)

        # Our end to end model
        end_to_end_model = tf.keras.Model(inputs, outputs)
        end_to_end_model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        # Test it with `raw_test_ds`, which yields raw strings
        end_to_end_model.evaluate(self.raw_test_ds)

        sample_text = ('The movie was cool. The animation and the graphics '
                       'were out of this world. I would recommend this movie.')
        sentiment = end_to_end_model.predict(np.array([sample_text]))

        # If the prediction is >= 0.0, it is positive else it is negative.
        print(sentiment)
