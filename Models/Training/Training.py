from Utils import early_stopping


def train_model(model, epochs, train_ds, val_ds, test_ds, es, batch_size):
    """
     ## Train the model
    """

    if es:
        es = early_stopping.es_callback
    else:
        es = None

    # Fit the model using the train and test datasets.
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=es, batch_size=batch_size)

    """
    ## Evaluate the model on the test set
    """
    model.evaluate(test_ds)

    return model
