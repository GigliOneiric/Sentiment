from Utils import utils


def train_model(model, epochs, train_ds, val_ds, test_ds, early_stopping):
    """
     ## Train the model
    """

    if early_stopping:
        early_stopping = utils.es_callback
    else:
        early_stopping = None

    # Fit the model using the train and test datasets.
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=early_stopping)

    """
    ## Evaluate the model on the test set
    """

    model.evaluate(test_ds)

    return model
