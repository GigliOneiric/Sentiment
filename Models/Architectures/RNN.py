import tensorflow as tf
from tensorflow.keras import layers



class RNN:

    def __init__(self, max_features, embedding_dim, Vectorization, raw_test_ds, test_ds, train_ds, val_ds):
        self.raw_test_ds = raw_test_ds
        self.test_ds = test_ds
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.max_features = max_features
        self.embedding_dim = embedding_dim

        self.Vectorization = Vectorization

    def create_rnn(self):
        # A integer input for vocab indices.
        inputs = tf.keras.Input(shape=(None,), dtype="int64")

        # Next, we add a layer to map those vocab indices into a space of dimensionality
        # 'embedding_dim'.
        x = layers.Embedding(self.max_features, self.embedding_dim)(inputs)
        x = layers.Dropout(0.5)(x)

        # Conv1D + global max pooling
        x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
        x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
        x = layers.GlobalMaxPooling1D()(x)

        # We add a vanilla hidden layer:
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

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
