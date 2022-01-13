from ..Models.logisticReg import LogisticRegression

class OneHiddenLayer():
    """
        n_x => size of input layer 
        n_h => size of hidden layer (this will be 4 since we need for now 4 hidden units)
        n_y => size of output layer
    """
    def layer_size_units(X, Y):
        n_x = X.shape[0]
        n_h = 4
        n_y = Y.shape[0]

        return n_x, n_h, n_y
    def init_parameters(n_x, n_h, n_y):
        pass 

    def forward_propagation(X, parameters):
        pass 

    def compute_cost(A2, Y):
        pass 

    def backward_propagation(parameters, cache, X, Y):
        pass 

    def update_parameters(parameters, grads, learning_rate=1.2):
        pass 

    def nn_model(X, Y, n_h, num_iteration=10000, print_cost=True):
        pass 

    def predict(parameters, X):
        pass 



