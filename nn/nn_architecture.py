from inspect import Parameter
from linecache import cache
import numpy as np
import matplotlib.pyplot as plt 
from model.activations import *

class Architecture():
    def __init__(self) -> None:
        pass
    # 2 - Layer NN
    def para_init_random(self, n_x, n_h, n_y):
        W1 = np.dot(np.random.randn(n_h, n_x), 0.01)
        b1 = np.zeros((n_h, 1))
        W2 = np.dot(np.random.randn(n_y, n_h), 0.01)
        b2 = np.zeros((n_y, 1))
        parameters = {
            "W1" : W1,
            "b1" : b1, 
            "W2" : W2,
            "b2" : b2
        }

        return parameters

    # L - Layers NN

    def para_init_random_deep(self, layer_dims):
        np.random.seed(1)
        parameters = {}
        num_layer = len(layer_dims)

        for l in range(1, num_layer):
            parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
            parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))

            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters

    """
        Z = WA + B # For linear Forward
    """
    def linear_forward(self, input_a, Weight, bias):
        Z = np.dot(Weight, input_a) + bias
        assert(Z.shape == (Weight.shape[0], input_a.shape[1]))
        cache = (input_a, Weight, bias)

        return Z, cache

    """
        Linear Activation Function Forward 
    """
    def linear_activation_forward(self, A_prev, Weight, bias, activation = 'sigmoid'):
        if (activation == 'sigmoid'):
            Z, linear_cache = self.linear_forward(A_prev, Weight, bias)
            A, activation_cache = sigmoid(Z)

        elif activation == 'relu':
            Z, linear_cache = self.linear_forward(A_prev, Weight, bias)
            A, activation_cache = relu(Z)
        assert (A.shape == (Weight.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def linear_backward(self, dZ, cache):
        #Cache are the returned from Forward_prop
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1/m * (np.dot(dZ, A_prev.T))
        db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation = 'sigmoid'):
        # Cache from computing backprop
        linear_cache, activation_cache = cache

        if (activation == 'relu'):
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        elif activation == 'sigmoid':
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db
        