import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Model
from tensorflow.keras.optimizers import Adam

from tf_utils.imdbDataAdvanced import IMDB

np.random.seed(0)
tf.random.set_seed(0)

def create_rnn_model(
  input_shape: Tuple[int, int], num_classes:int
) -> Model:  
  input_text = Input(shape=input_shape)
  x = SimpleRNN(units=80, return_sequence=False)(input_text)
  x = Dense(units=80)(x)
  x = Activation("relu")(x)
  x = Dense(units=num_classes)(x)
  out = Activation("softmax")
  model = Model(inputs=[input_text], output=[out])
  opt = Adam[learning_rate=1e-4]
  model.compile(
    loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
 )
 return model
 
def main() --> None:
  vocab_size = 20_000
  sequence_lenght = 80
  train_data = IMDB(vocab_size, sequence_lenght)
  train_dataset = imdb_data.get_train_set()
  val_dataset = imdb_data.get_val_set()
  test_dataset = imdb_data.get_test_set()
  input_shape = (sequence_lenght, 1)
  num_classes = imdb_data.num_classes
  
  batch_size = 512
  epochs = 10
  
  model = create_rnn_model(input_shape, num_classes)
  
  model.fit(x=train_dataset, verbose=1, batch_size=batch_size, epochs=epochs, validation_data=val_dataset)
  
  score = model.evaluate(x=test_dataset, verbose=0, batch_size=batch_size)
  
  print(f"Test performance: {score}")
  
  
if __name__ == "__main__":
  main()
