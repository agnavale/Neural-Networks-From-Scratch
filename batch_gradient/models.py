import numpy as np
import utils

class Sequential:
    def __init__(self,layers = []):
        self.layers = layers
        self.loss = None
        self.optimizer = None
        self.metric = None

    def add(self,layer):
        self.layers.append(layer)

    def compile(self, loss= "Cross_entropy", optimizer = "SGD", metric= None):
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        
    def predict(self,input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self,x_train, y_train, batch_size = 1, epochs = 1, verbose='true'):
        #batch gradient decend
        x_batches = utils.create_batches(x_train,batch_size)
        y_batches = utils.create_batches(y_train,batch_size)
        
         # optimizer
        optimizer = utils.initiate_optimzer(self.optimizer)

        for e in range(epochs):
            error = 0
            for x, y in zip(x_batches, y_batches):
                # forward
                output = self.predict(x)

                # error
                loss = utils.initiate_loss(self.loss)
                error += loss.forward(y,output)
                grad = loss.backward(y,output)

                # backward
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
                    optimizer.update_parms(layer)
                
            error /= len(x_train)
            if verbose:
                if (e+1)%100 == 0:
                    print(f"{e + 1}/{epochs}, error={error}")
            
