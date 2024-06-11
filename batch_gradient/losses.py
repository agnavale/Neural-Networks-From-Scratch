import numpy as np

class Loss:
    def forward(self, y_true, y_pred):
        # TODO: returns mean loss
        pass

    def backward(self, y_true, y_pred):
        # TODO: returns loss_prime vector
        pass

# losses
class MSE(Loss):
    def forward(self,y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self,y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true, axis= 1)

class Cross_entropy(Loss):
    def forward(self,y_true, y_pred):
        y_pred_clipped = np.clip(y_pred, 1e-8,1-1e-8)
        return np.mean(-y_true * np.log(y_pred_clipped))

    def backward(self,y_true, y_pred):
        y_pred_clipped = np.clip(y_pred, 1e-2,1-1e-2)
        return (-y_true/ y_pred_clipped) / np.size(y_true)



