from Models.logisticReg import LogisticRegression
from datasets_code.load_dataset import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset() 
def find_image_values():
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    return m_train, m_test, num_px

def flatten_resize_image():
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # Check that the first 10 pixels of the second image are in the correct place
    assert np.alltrue(train_set_x_flatten[0:10, 1] == [196, 192, 190, 193, 186, 182, 188, 179, 174, 213]), "Wrong solution. Use (X.shape[0], -1).T."
    assert np.alltrue(test_set_x_flatten[0:10, 1] == [115, 110, 111, 137, 129, 129, 155, 146, 145, 159]), "Wrong solution. Use (X.shape[0], -1).T."

    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    return train_set_x, test_set_x, train_set_x_flatten, test_set_x_flatten


def main():
    logisticReg = LogisticRegression()
    m_train, m_test, num_px = find_image_values()
    x_train, x_test, train_x_flatten, test_x_flatten = flatten_resize_image()

    test_sigmoid()
    print("init: ")
    test_initialization()
    print("propagate")
    test_propagate()

    logistic_regression_model = logisticReg.model(
                                                train_x_flatten, 
                                                train_set_y, 
                                                test_x_flatten, 
                                                test_set_y, 
                                                num_iterations=2000, 
                                                learning_rate=0.005, 
                                                print_cost=True)

    test_my_own_image("cat001.jpg", logistic_regression_model)


def test_sigmoid():
    logisticReg = LogisticRegression()
    x = np.array([0.5, 0, 2.0])
    output = logisticReg.sigmoid(x)
    print(output)


def test_initialization():
    logisticReg = LogisticRegression()
    dim = 2
    w, b = logisticReg.init_parameters_with_zeros(dim)

    assert type(b) == float

    #initialize_with_zeros_test(initialize_with_zeros)

def test_propagate():
    logisticReg = LogisticRegression()
    w =  np.array([[1.], [2]])
    b = 1.5
    X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
    Y = np.array([[1, 1, 0]])
    cost, grads = logisticReg.propagate(w, b, X, Y)


    assert type(grads["dw"]) == np.ndarray
    assert grads["dw"].shape == (2, 1)
    assert type(grads["dbias"]) == np.float64


# this is for testing my own image 
def test_my_own_image(my_image, logistic_regression_model):
    # change this to the name of your image file
    logisticReg = LogisticRegression()
    m_train, m_test, num_px = find_image_values()

    # We preprocess the image to fit your algorithm.
    fname = "images/" + my_image
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    plt.imshow(image)
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T
    my_predicted_image = logisticReg.predict(logistic_regression_model["weight"], logistic_regression_model["bias"], image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


def check_learning_rate(x_train, x_test):
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    logisticReg = LogisticRegression()
    for lr in learning_rates:
        print ("Training a model with learning rate: " + str(lr))
        models[str(lr)] = logisticReg.model(x_train, train_set_y, x_test, test_set_y, num_iterations=1500, learning_rate=lr, print_cost=False)
        print ('\n' + "-------------------------------------------------------" + '\n')
    for lr in learning_rates:
        plt.plot(np.squeeze(models[str(lr)]["cost"]), label=str(models[str(lr)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
if __name__ == "__main__":
    main()
