from linecache import cache
from turtle import forward, update
import numpy as np
from model.activations import *
import copy
from model.nn_arch import * 


class Single_NN:

    # Define Layer size:
    def layer_sizes(self, X, Y, num_hidden_units = 4):
        n_x = X.shape[0] # Input layer #Units.
        n_h = num_hidden_units # Hidden Layer => 4 Units
        n_y = Y.shape[0] # Output layer #Units

        return n_x, n_h, n_y

    """
        Forward Propagation: Will compute Z based on Z = wx + b , and than a = activation(z) for each layer. z1 = w1, b1, z2 = w2, b1
        z1 = w1*input_X + b1
        a1 = activation(z1)
        z2 = w2*a1 + b2
        a2 = activation(z2) ..etc.
    """
    def forward_propagation(self, X, parameters):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        #First let's try: 
            # Relu(z1) => Relu(z2) => Sigmoid(Z3)
        Z1 = np.dot(W1, X)+b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1)+b2
        A2 = sigmoid(Z2)

        cache = {
            "Z1":Z1,
            "A1":A1,
            "Z2":Z2,
            "A2":A2
        }

        return A2, cache


    """
        Backward Propagation: 
            * X - input features, Y- target output, cache - Everything from forward propagation.
            * Starts to apply dervitive on a3 first like going backward from output layer activation function. (using gradient descent)
            * 
    """
    def backward_propagation(self, parameters, X, Y, cache):
        m = X.shape[1]

        # Get Weights since we have singular hidden layer, we will have only two weights.
        W1 = parameters["W1"]
        W2 = parameters["W2"]

        # Retrieve A1 and A2 from cache.
        A1 = cache["A1"]
        A2 = cache["A2"]

        #Backprop starts here: for singular layer.
        dZ2 = np.subtract(A2, Y) # dz2 = A2 - Y
        dW2 = 1/m * np.dot(dZ2, A1.T)
        db2 = 1/m*np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.multiply(np.dot(W2.T, dZ2), (1-np.power(A1, 2)))
        dW1 = 1/m*np.dot(dZ1, X.T)
        db1 = 1/m* np.sum(dZ1, axis=1, keepdims = True)



        grads = {
            "dW1":dW1,
            "db1":db1,
            "dW2":dW2,
            "db2":db2
        }

        return grads

    def update_parameters(self, parameters, grads, learning_rate=0.01):
        #retrieve W1, b1, W2, b2
        W1 = copy.deepcopy(parameters["W1"])
        b1 = copy.deepcopy(parameters["b1"])
        W2 = copy.deepcopy(parameters["W2"])
        b2 = copy.deepcopy(parameters["b2"])

        # Get derivitives: 
        dW1 = copy.deepcopy(grads["dW1"])
        db1 = copy.deepcopy(grads["db1"])
        dW2 = copy.deepcopy(grads["dW2"])
        db2 = copy.deepcopy(grads["db2"])

        # Update the parameters W1, b1, W2, b2
        W1 = W1 - np.multiply(learning_rate, dW1)
        b1 = b1 - np.multiply(learning_rate, db1)
        W2 = W2 - np.multiply(learning_rate, dW2)
        b2 = b2 - np.multiply(learning_rate, db2)

        return {
            "W1":W1,
            "b1":b1,
            "W2":W2,
            "b2":b2
        }

    def compute_cost(self,A2, Y):
        m = Y.shape[1]
        # this will be fixed by adding adams algorithm.s
        logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1-A2))
        cost = -1/m*np.sum(logprobs)
        
        cost = float(np.squeeze(cost)) # Make sure we get the dimension we expect => [[17]] => 17
        return cost

    def nn_model(self, X, Y, n_hidden_units = 4, num_iteration=10000, print_cost = True):
        np.random.seed(3)
        # Get layers sizes. 
        n_x = self.layer_sizes(X, Y)[0]
        n_y = self.layer_sizes(X, Y)[2]

        # Initialize Parameters. Try with he_init, than try with random_init. Give users ability to choose later.
        layer_dims = [n_x, n_hidden_units, n_y]
        parameters = he_initialization(layer_dims)

        for i in range(0, num_iteration): # Loop over number of iteration
            A2, cache = self.forward_propagation(X, parameters)
            cost = self.compute_cost(A2, Y)
            gradients = self.backward_propagation(parameters, X, Y, cache)
            parameters = self.update_parameters(parameters, gradients)

            if (print_cost and i%1000 == 0):
                print("cost after iteration %i : %f" %(i, cost))

        return parameters


    def predict(self, parameters, X):
        # Compute probabilities using forward prop and classifies to 0/1 using 0.5 as the threshold.
        A2, cache = self.forward_propagation(X, parameters)
        predictions = (A2 > 0.5) # Base on two classification 0 || 1 WILL CHANGE LATER TO MULTI-CLASSIFICATION.

        return predictions