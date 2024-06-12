import numpy as np
import utils

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update weights_gradient and return input gradient
        pass

class Dense(Layer):
    def __init__(self, input_dims, units, activation ='Linear'):
        self.weights = np.random.randn(input_dims,units)
        self.bias = np.random.randn(1,units)
        self.activation = utils.initiate_activation(activation)
        self.weights_gradient = None
        self.bias_gradient = None
    
    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        act_output = self.activation.forward(self.output)
        return act_output

    def backward(self, output_gradient):
        output_gradient = self.activation.backward(output_gradient)
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        self.bias_gradient = np.mean(output_gradient, axis=0, keepdims= True)
        input_gradient = np.dot(output_gradient,self.weights.T)
        return input_gradient



