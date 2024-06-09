import numpy as np

import losses
import activations
import optimizers

'''utils contains functions that are commonly used in different modules'''

def initiate_optimzer(obj):
    if type(obj) == str:
        return getattr(optimizers,obj)()
    return obj

def initiate_loss(obj):
    if type(obj) == str:
        return getattr(losses,obj)()
    return obj

def initiate_activation(obj):
    if type(obj) == str:
        return getattr(activations,obj)()
    return obj