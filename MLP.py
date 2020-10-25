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
    __init__ takes 3 arguments and initializes attributes:
      1) seed, seed for random
      2) n_layer_nodes, the number of nodes per layer - also used to find the number of hidden layers; first value matches dimension of inputs
      3) verbose, flag
  """

  def __init__(
    self,
    seed=None,
    n_layer_nodes=[3,3,3,3],
    verbose=False
  ):
    np.random.seed(seed)
    self.verbose = verbose
    self.pred_training = []
    self.truth_training = []
    self.error_record = []
    # instantiate layers
    # the layers attribute holds Layer objects
    # each Layer object has an attribute representing the
    # weights leading into the layer, including the output layer.
    # the layer is responsible for calculating and activating
    # its own output, and for performing its own backprop
    self.layers = [
      Layer(
        n_layer_nodes[i-1],
        n_layer_nodes[i],
        seed
      )
        for i in range(1, len(n_layer_nodes))
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
    train takes 5 arguments and uses backprop to update the weights based on the training data, returns self for chaining:
      1) training_inputs, ndarray of training feature space
      2) labels, 2d-ndarray of classes in one-hot-encoded representation
      3) epochs
      4) learning_rate
      5) momentum
  """

  def train(
    self,
    training_inputs,
    outputs,
    epochs=1,
    learning_rate=0.1,
    momentum=0.1
  ):
    for i in range(epochs): # loop epochs
      if self.verbose:
        print('Epoch', i + 1, 'started.')
      trial = 0
      for inputs, output in zip(training_inputs, outputs): # loop records
        if self.verbose:
          trial += 1
          if trial % 1000 == 0:
            print('    Trial', trial, '/', outputs.size, 'running')
        prediction = self.predict_record(inputs)
        if self.verbose:
          pred_class = np.argmax(prediction)
          truth_class = np.argmax(output)
          print('Pred:', pred_class, 'Actual:', truth_class)
        self.pred_training.append(prediction)
        self.truth_training.append(output)
        delta = output - prediction
        error = self.error(delta) # calculate SSE
        self.error_record.append(error)
        if self.verbose:
          print('    Trial', trial, 'error:', error)
        if error != 0:
          for layer in reversed(self.layers): # loop layers backwards for backprop
            delta = layer.backprop(delta, learning_rate, momentum)

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
      3) seed, seed for random
  """

  def __init__(
    self,
    dim_prev,
    dim,
    seed=None
  ):
    np.random.seed(seed)
    self.weights = np.random.rand(dim_prev + 1, dim) * 0.0001
    self.previous = np.zeros((dim_prev + 1, dim)) # for momentum
    self.inputs = np.zeros(dim_prev + 1)
    self.outputs = np.zeros(dim)

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
    backprop takes 3 arguments and updates weights, returns delta for next layer:
      1) delta, ndarray of node-output delta
      2) learning_rate, allows dynamic changes throughout learning
      3) momentum, allows dynamic changes throughout learning
  """

  def backprop(
    self,
    delta,
    learning_rate,
    momentum
  ):
    errorTerm = delta * self.outputs * (1 - self.outputs)
    change = learning_rate * np.outer(self.inputs, errorTerm) + momentum * self.previous
    self.weights += change
    self.previous = change
    return(np.sum(errorTerm * self.weights[1:], axis=1))

  def sigmoid(
    self,
    x_arr
  ):
    return(1 / (1 + np.exp(-x_arr)))
