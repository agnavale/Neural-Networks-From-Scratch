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

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        self.output =  self.activation(self.input)
        return self.output
        
    def backward(self, output_gradient):
        return np.multiply(output_gradient, self.activation_prime(self.input))

# activation functions
class Linear(Activation):
    def __init__(self):
        def linear(x):
            return x

        def linear_prime(x):
            return np.ones(x.shape)

        super().__init__(tanh, tanh_prime)

class Relu(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0,x)

        def relu_prime(x):
            return (x>0)*1

        super().__init__(relu, relu_prime)

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)



class Softmax(Layer):
    def forward(self, input):
        # tmp = np.exp(input)- np.max(input, axis=1, keepdims=True)
        tmp = np.exp(input-np.max(input))
        self.output = tmp / np.sum(tmp, axis=1,keepdims=True)
        
        # print("act:",self.output)
        return self.output
    
    def backward(self, output_gradient):
        # This version is faster than the one presented in the video
        input_gradient = np.empty_like(output_gradient)
        
        for i, (single_output, single_output_gradient) in enumerate(zip(self.output,output_gradient)):
            
            single_output = single_output.reshape(-1,1)
            
            jacobMatr = np.diagflat(single_output) - \
                        np.dot(single_output,single_output.T)
            
            input_gradient[i] = np.dot(jacobMatr,single_output_gradient)

        return input_gradient
