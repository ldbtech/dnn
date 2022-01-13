import numpy as np 
import pickle
import copy

class LogisticRegression():

    def __init__(self) -> None:
        pass

        """
            # Activation functions here => Sigmoid, tanh, leaky_relu, softmax 
        """
    def sigmoid(self, Z):
        s = 1/(1+np.exp(-1*Z))
        return s

    def tanh(self, Z):
        tanh_func = (np.exp(Z)-np.exp(-Z)) / (np.exp(Z)-np.exp(-Z))
        return tanh_func

    def leaky_relu(self, Z, alpha):
        return max(alpha*Z, Z)

    # Function definition: e^zi / (sum(e^zj))
    def softmax(self, Z):
        exp = np.exp(Z)
        return exp/np.sum(exp)

        """
            Initialize paramaters. 
        """
    def init_parameters_with_zeros(self ,dim):
        weight = np.zeros((dim, 1))
        bias  =  0.0
        return weight, bias

        """
            Forward Propagation & Backward Propagation:
        """
    def propagate(self, weight, bias, X, Y, activation='sigmoid', alpha = 0.01):
        """
        Arguments:
            w -- weights, a numpy array of size (num_px * num_px * 3, 1)
            b -- bias, a scalar
            X -- data of size (num_px * num_px * 3, number of examples)
            Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
            cost -- negative log-likelihood cost for logistic regression 
            dw -- gradient of the loss with respect to w, thus same shape as w
            db -- gradient of the loss with respect to b, thus same shape as b
        """
        m = X.shape[1]
        # Forward Propagation
        if (activation == 'sigmoid'):
            A = self.sigmoid(np.dot(weight.T, X)+bias)
        elif(activation == 'tanh'):
            A = self.tanh(np.dot(weight.T, X)+bias)
        elif(activation == "softmax"): 
            A = self.softmax(np.dot(weight.T, X)+bias)
        else:
            A = self.leaky_relu((np.dot(weight.T, X)+bias), alpha)
        cost = -1/m*(np.dot(Y, np.log(A).T)+np.dot((1-Y), np.log(1-A).T))
       # cost = -1/m * (np.dot(Y, np.log(A))) + np.dot((1-Y), np.log(1-A).T)

        #Backward Propagation
        # Implement gradient descent. 
        dweight = 1/m * np.dot(X, (A-Y).T)
        dbias = 1/m * np.sum(A-Y)

        cost = np.squeeze(np.array(cost))

        grads = {'dw':dweight, 'dbias':dbias}

        return cost, grads

    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

    Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
        You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    def optimization(self, weight, bias, X, Y, num_iterations=100, learning_rate = 0.009, print_cost = True, activation = 'sigmoid'):
      
        weight = copy.deepcopy(weight)
        bias = copy.deepcopy(bias)
        
        costs = []

        for i in range(num_iterations):
            cost, grads = self.propagate(weight, bias, X, Y, activation=activation)

            # dervitives: 
            dw = grads['dw']
            db = grads['dbias']

            # Update Weight, bias: 
            weight = weight - learning_rate*dw
            bias = bias - learning_rate*db

            #Record cost whenever i devides 100
            if (i % 100 == 0):
                costs.append(cost)
                # print the cost every 100 training iterations 
                if print_cost:
                    print('Cost after iteration %i: %f'%(i, cost))
        params = {"weight":weight, "bias" : bias}
        grads = {"dw":dw, "dbias":db}

        return params, grads, costs

    # this is Y-hat since Y-hat have to be always sigmoid so far we learned: 
        # Y-hat = A = sigmoid(W.T*X+bias)
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        
     Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    def predict(self, weight, bias, X):

        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        weight = weight.reshape(X.shape[0], 1)
            
        A = self.sigmoid(np.dot(weight.T, X)+bias)
        for i in range(A.shape[1]):
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            #(≈ 4 lines of code)
            # if A[0, i] > ____ :
            #     Y_prediction[0,i] = 
            # else:
            #     Y_prediction[0,i] = 
            # YOUR CODE STARTS HERE
            if (A[0, i] > 0.5):
                Y_prediction[0, i] = 1
            else:
                Y_prediction[0, i] = 0
        return Y_prediction

    def model(self, X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False, activation='sigmoid'):
        # First initiliazing Weight, bias 
        # Optimize params, grads , costs 
        #Y_prediction_test, Y_prediction_train, predict(X_test), predict(X_train)

        w, b = self.init_parameters_with_zeros(X_train.shape[0])
        param, grads, costs = self.optimization(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
        w = param['weight']
        b = param['bias']
        
        Y_prediction_test = self.predict(w, b, X_test)
        Y_prediction_train = self.predict(w, b, X_train)

        if (print_cost):
            print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
            print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        
        d = {
            "cost": costs, 
            "Y_prediction_train":Y_prediction_train,
            "Y_prediction_test":Y_prediction_test,
            "weight":w, 
            "bias":b,
            "learning_rate":learning_rate, 
            "num_iterations":num_iterations
        }

        return d





        









