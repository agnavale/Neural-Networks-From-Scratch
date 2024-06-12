import numpy as np

class SGD:
    def __init__(self,learning_rate = 0.1):
        self.learning_rate = learning_rate

    def update_parms(self, layer):
        layer.weights -= self.learning_rate * layer.weights_gradient
        layer.bias -= self.learning_rate * layer.bias_gradient
        
class Momentum:
    def __init__(self, learning_rate = 0.1, beta = 0.9):
        self.learning_rate = learning_rate
        self.beta = beta

    def update_parms(self,layer):
        if not hasattr(layer, "weights_velocity"):
                layer.weights_velocity = np.zeros(np.shape(layer.weights))
                layer.bias_velocity = np.zeros(np.shape(layer.bias))

        layer.weights_velocity = self.beta * layer.weights_velocity + (1-self.beta)*layer.weights_gradient
        layer.bias_velocity = self.beta * layer.bias_velocity + (1-self.beta)*layer.bias_gradient

        layer.weights -= self.learning_rate * layer.weights_velocity
        layer.bias -= self.learning_rate *  layer.bias_velocity
       
       
