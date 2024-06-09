import numpy as np

class SGD:
    def __init__(self,learning_rate = 0.1):
        self.learning_rate = learning_rate

    def update_parms(self, layer):
        if hasattr(layer, "weights"):
            layer.weights -= self.learning_rate * layer.weights_gradient
            layer.bias -= self.learning_rate * layer.bias_gradient
        
class Momentum:
    def __init__(self, learning_rate = 0.1, momentum = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update_parms(self,layer):
        if hasattr(layer, "weights"):
            if self.momentum:
                if not hasattr(layer, "weights_momentum"):
                    layer.weights_momentum = np.zeros(np.shape(layer.weights))
                    layer.bias_momentum = np.zeros(np.shape(layer.bias))

                weight_updates = self.learning_rate * layer.weights_gradient - self.momentum * layer.weights_momentum
                bias_updates = self.learning_rate * layer.bias_gradient - self.momentum * layer.bias_momentum
            else:
                weight_updates = self.learning_ratening * layer.weights_gradient
                bias_updates = self.learning_rate * layer.bais_gradient

            layer.weights -= self.learning_rate * weight_updates
            layer.bias -= self.learning_rate * bias_updates