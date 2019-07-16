import numpy as np


def Q_values(x, W1, W2, bias_W1, bias_W2):

    # Neural activation: input layer -> hidden layer
    out1 = np.maximum(x.dot(W1.T) + bias_W1, 0) 
    # Neural activation: hidden layer -> output layer
    Q = np.maximum(out1.dot(W2.T) + bias_W2, 0) 
    
    return Q, out1