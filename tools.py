import numpy as np
import matplotlib.pyplot as plt


def visualize_number(a_vector):
    """ Takes a Numpy array vector of MNIST
        data and visualizes it.
    """

    # Reshaping the vector as 28x28 image, so it can
    # be rendered.
    pixels = a_vector.reshape(28, 28)

    plt.title("Number")
    plt.imshow(pixels, cmap="gray_r")
    plt.show()


def sigmoid(x):
    """ A sigmoid activation function to be
        used with a neural network implementation

        We're using np.exp because if x is a vector, then
        numpy applies it as elementwise.
    """
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    """ Derivation sigmoid(x) """
    return sigmoid(x) * (1 - sigmoid(x))

def hyperbolic(x):
    """ Hyperbolic tangent activation function
        to be used with a neural network implementation.
    """

    return (np.exp(2*x) - 1.0) / (np.exp(2*x) + 1.0)

