"""
  Title: Backprop
  Class: CISC/CMPE 452/COGS 400
  Description: A class implementing an MLP predictor with momentum-based backprop
"""

import numpy as np

"""
  The MLP class implements the classifier
"""

class MLP(object):

  """
    __init__ takes 5 arguments and initializes attributes:
      1) epochs, the number of iterations through the training data
      2) learning_rate, c
      3) seed, seed for random
      4) n_layer_nodes, the number of nodes per layer - also used to find the number of hidden layers; first value matches dimension of inputs
      5) momentum, alpha
      6) verbose, flag
  """

  def __init__(
    self,
    seed=None,
    epochs=300,
    learning_rate=0.1,
    n_layer_nodes=[3,3,3,3],
    momentum=0.1,
    verbose=False
  ):
    np.random.seed(seed)
    self.epochs = epochs
    self.verbose = verbose
    n_layer_nodes.insert(0, n_layer_nodes[0])
    self.layers = [
      Layer(
        n_layer_nodes[i-1],
        n_layer_nodes[i],
        learning_rate,
        momentum,
        seed
      )
        for i in range(len(n_layer_nodes))
        if i != 0
    ]

  """
    predict_record takes 1 argument and returns a prediction for the record:
      1) inputs, an ndarray representing the feature space for one record
  """

  def predict_record(
    self,
    inputs
  ):
    for layer in self.layers:
      inputs = layer.output(self.appendBias(inputs))

    return(inputs)

  """
    train takes 2 arguments and uses backprop to update the weights based on the training data, returns self for chaining:
      1) training_inputs, ndarray of training feature space
      2) labels, 2d-ndarray of classes in one-hot-encoded representation
  """

  def train(
    self,
    training_inputs,
    outputs
  ):
    for i in range(self.epochs):
      if self.verbose:
        print('Epoch', i, 'started.')
      trial = 0
      for inputs, output in zip(training_inputs, outputs):
        if self.verbose:
          trial += 1
          if trial % 25000 == 0:
            print('    Trial', trial, '/', outputs.size, 'running')
        prediction = self.predict_record(inputs)
        delta = output - prediction
        error = self.error(delta)
        if error != 0:
          for layer in reversed(self.layers):
            delta = layer.backprop(delta)

    return(self)

  """
    error takes 1 argument and returns the SSE:
      1) delta, ndarray of deltas
  """

  def error(
    self,
    delta
  ):
    return(np.sum(delta ** 2))

  """
    predict takes 1 argument and returns an ndarray with the predicted classes:
      1) prediction_inputs, the feature space of all records to predict
  """

  def predict(
    self,
    prediction_inputs
  ):
    return(np.array([
      self.predict_record(inputs)
      for inputs in prediction_inputs
    ]))

  def appendBias(
    self,
    inputs
  ):
    return(np.insert(inputs, 0, 1., axis=0))

"""
  The Layer class implements a non-input layer, the weights feeding into the layer, backprop, and a method to generate the output from the layer
"""

class Layer(object):

  """
    __init__ takes 3 arguments and initializes attributes:
      1) dim_prev, the number of nodes in the previous layer
      2) dim, the number of nodes in the current layer
      3) learning rate, c
      4) momentum, alpha
      5) seed, seed for random
  """

  def __init__(
    self,
    dim_prev,
    dim,
    learning_rate=0.1,
    momentum=0.1,
    seed=None
  ):
    np.random.seed(seed)
    self.weights = np.random.rand(dim_prev + 1, dim)
    self.previous = np.zeros((dim_prev + 1, dim)) # for momentum
    self.inputs = np.zeros(dim_prev + 1)
    self.outputs = np.zeros(dim)
    self.learning_rate = learning_rate
    self.momentum = momentum

  """
    output takes 1 argument and returns an output for the layer:
      1) inputs, an ndarray representing the input feature space
  """

  def output(
    self,
    inputs
  ):
    self.inputs = inputs
    self.outputs = self.sigmoid( np.dot(inputs, self.weights) )
    return(self.outputs)

  """
    backprop takes 2 arguments and updates weights, returns delta for next layer:
      1) delta, ndarray of node-output delta
  """

  def backprop(
    self,
    delta
  ):
    errorTerm = np.tile(np.transpose(delta * self.outputs * (1 - self.outputs)), (self.inputs.size, 1))
    change = self.learning_rate * errorTerm * np.transpose(np.tile(self.inputs, (self.outputs.size, 1))) + self.momentum * self.previous
    self.weights += change
    self.previous = change
    return(np.sum(errorTerm[1:] * self.weights[1:], axis=1))

  def sigmoid(
    self,
    x_arr
  ):
    return(1 / (1 + np.exp(-x_arr)))
