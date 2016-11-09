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




