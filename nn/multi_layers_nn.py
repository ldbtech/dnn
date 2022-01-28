from mimetypes import init
from platform import architecture
from model.nn_arch import init_weight_bias_random
import numpy as np
from .nn_architecture import Architecture as arch
import copy
class MultiLayerNetwork():        
    def layer_model_forward(self, X, parameters):
        caches = []
        A = X
        architecture = arch()
        num_layers = len(parameters) // 2

        for l in range(1, num_layers):
            A_prev = A
            A, cache = architecture.linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], "relu")
            caches.append(cache)

        
        # Last Layer which is the output layer will be always sigmoid for now. 
        # AL activation of the last layer needed for backprop
        AL, cache = architecture.linear_activation_forward(A, parameters['W'+str(num_layers)], parameters['b'+str(num_layers)], "sigmoid")
        caches.append(cache)

        return AL, caches
    
    def compute_cost(self, AL, Y):
        m = Y.shape[1]

        cost = -1/m * (np.sum(Y*np.log(AL) + (1-Y)*np.log(1 - AL)))

        cost = np.squeeze(cost)

        return cost

    def layer_model_backward(self, AL, Y, caches):
        architecture = arch()
        grads = {}
        num_layer = len(caches) # Number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # After this Y.shape == AL.shape

        # Init _ Backprop
        dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))

        current_cache = caches[num_layer - 1]

        grads["dA"+str(num_layer - 1)], grads["dW"+str(num_layer)], grads["db"+str(num_layer)] = architecture.linear_activation_backward(dAL, current_cache, "sigmoid")

        for l in reversed(range(num_layer-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = architecture.linear_activation_backward(grads['dA'+str(l+1)], current_cache, "relu")
            grads["dA"+str(l)], grads["db"+str(l+1)], grads["dW"+str(l+1)]  = dA_prev_temp, db_temp, dW_temp

        return grads

    def update_parameters(self, params, grads, learning_rate):
        parameters = copy.deepcopy(params)
        num_layer = len(parameters) // 2 # Number of layers in nn

        for l in range(num_layer):
            parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
            parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*grads['db'+str(l+1)]
        return parameters
         

    def L_layer_model(self, X, Y, layer_dims, learnig_rate = 0.0074, num_iterations = 3000, print_cost = True):
        architecture = arch()
        np.random.seed(1)
        costs = []

        # Init parameters 
        parameters = architecture.para_init_random_deep(layer_dims)

        for i in range(0, num_iterations):
            AL, caches = self.layer_model_forward(X, parameters)

            cost = self.compute_cost(AL = AL, Y = Y)

            grads = self.layer_model_backward(AL, Y, caches)

            parameters = self.update_parameters(parameters, grads, learnig_rate)

            if(print_cost and i % 100 == 0 or i == num_iterations-1):
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost))) 
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)
        return parameters, costs

    def predict(self, X, y, parameters):
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = self.layer_model_forward(X, parameters)

        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        print("Accuracy: "  + str(np.sum((p == y)/m)))
            
        return p


