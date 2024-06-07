import numpy as np
import losses


class Sequential:
    def __init__(self,layers = []):
        self.layers = layers
        self.loss = None

    def add(self,layer):
        self.layers.append(layer)

    def compile(self, loss= "cross_entropy", optimiser = None, metric= None):
        self.loss = loss
        
            
    def predict(self,input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self,x_train, y_train, epochs, learning_rate, verbose='true'):
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                # forward
                output = self.predict(x)

                # error
                loss = getattr(losses, self.loss)
                loss_prime = getattr(losses,self.loss+"_prime")

                error += loss(y,output)

                # backward
                grad =loss_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)
            if verbose:
                print(f"{e + 1}/{epochs}, error={error}")
            
  
