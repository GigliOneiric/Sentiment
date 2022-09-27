from tensorflow.keras.callbacks import EarlyStopping

es_callback = EarlyStopping(
    monitor="val_loss", patience=5, verbose=1, restore_best_weights=True
)
