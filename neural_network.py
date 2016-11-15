import numpy as np
import tools as t

class NeuralNetwork(object):
    bias_filename = "biases.npy"
    weights_filename = "weights.npy"

    def __init__(self, layers):
        """ Initialization.

            layers:     List of layer neurons. [2 3 1]

            Initializes weights and biases randomly using Numpy's
            random array generator.
        """

        self.number_of_layers = len(layers)
        self.layers = layers

        # Biases.
        # Each neuron in each layer has its own bias, but the biases in
        # input layer (layers[0]) are ignored because, biases in general
        # are used for calculating the output of the neurons.

        # We basically create a vector via transpose.
        # If a layer has 3 neurons, randn(3,1) creates a 3x1 matrix, aka a vector.
        # Which represented in Python in the following way:
        # [[0.1],
        #  [0.2],
        #  [0.1]]

        # biases[0] -> biases in layers[1]
        self.biases = [np.random.randn(x, 1) for x in layers[1:]]

        # Similarly we create a transposed weights lists.
        # weights[0] -> weights connecting layers[0] -> layers[1]

        # Here we create a list of tuples then represent the weights connections.
        # Example:  layers = [A B C] where A, B, C are integers.
        #           zipped = [(A, B), (B, C)]
        zipped = zip(layers[:-1], layers[1:])

        self.weights = [np.random.randn(y, x) for x, y in zipped]
        # Example:  layers = [2, 3, 1]
        #           weights = [ array([[ 0.75054289, -0.18324735],
        #                               [ 0.32141292, -0.54226539],
        #                               [-0.53575605,  0.25761202]]),
        #                       array([[ 0.22304971,  1.29879581, -0.49437018]])]

    def save_state(self):
        """ Save the current state of the network.

            Might be useful for performance and time limitations reasons.
            When a training is stopped before it was completed, its states can be
            saved and loaded later on.
        """

        np.save(self.bias_filename, self.biases)
        np.save(self.weights_filename, self.weights)

    def load_state(self):
        """ Loads state from save *npy file.
        """

        b = np.load(self.bias_filename)
        self.biases = b.tolist()

        w = np.load(self.weights_filename)
        self.weights = w.tolist()

    def feed_forward(self, a):
        """ a is our input, and it's in the form of columnn vector.
            Just as np.random.randn(3,1) would produces a (3x1)
            column vector.
        """

        # With each step, activation of layer is updated to represent
        # the activation of the next layer.
        # For a [2,3,1] neural network.
        #   Step 1: Input to Hidden Layer   w(3x2) x a(2x1) + b(3x1) -> sigmoid -> a' (3x1)
        #   Step 2: Hidden Layer to Output  w(1x3) x a(3x1) + b(1x1) -> sigmoid -> a' which is the output.
        for b, w in zip(self.biases, self.weights):
            a = t.sigmoid(np.dot(w, a) + b)

        return a



nn = NeuralNetwork([2, 3, 1])
rand_inp = np.random.randn(2, 1)

print nn.feed_forward(rand_inp)
