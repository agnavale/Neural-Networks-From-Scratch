import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass

class Dense(Layer):
    def __init__(self,input_size,output_size):
        self.weights = np.random.randn(input_size,output_size)
        self.bias = np.random.randn(output_size,1)
    
    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights.T, self.input) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input, output_gradient.T)
        input_gradient = np.dot(self.weights, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient



