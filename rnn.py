import numpy as np
from tensorflow import keras


class Rnn:
  def __init__(self, input_size, epochs = 100, batch_size = 100):
    self.model = keras.models.Sequential([
    keras.layers.SimpleRNN(input_size, return_sequences = True, input_shape=[input_size, 1]),
    keras.layers.SimpleRNN(input_size, return_sequences = True),
    keras.layers.SimpleRNN(1)
])
    self.epochs = epochs
    self.batch_size = batch_size

  def compile(self, optimizer = 'adam', loss = 'mse'):
    self.model.compile(optimizer=optimizer,
                loss=loss,
                metrics=['accuracy'])
    
  def train(self, x_train, y_train, x_valid, y_valid):
    return self.model.fit(x_train, y_train, epochs = self.epochs, batch_size = self.batch_size,
               validation_data=(x_valid, y_valid), verbose = 0)
    
  def simple_predict(self, x_predict):
    return self.model(x_predict)
  
  def predict_few_days(self, x_predict, days):
    result = []
    inp = x_predict
    for _ in range(days):
      result.append(self.model(inp))
      inp = np.delete(inp, 0, 1)
      val = [result[-1]]
      inp = np.append(inp, val, 1)
    return result
