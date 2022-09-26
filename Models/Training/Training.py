def train_model(model, epochs, train_ds, val_ds, test_ds):
    """
     ## Train the model
    """

    # Fit the model using the train and test datasets.
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    """
    ## Evaluate the model on the test set
    """

    model.evaluate(test_ds)

    return model
