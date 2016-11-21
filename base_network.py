import numpy as np
import tools as t


class BaseNetwork(object):
    state_folder = "states/"

    bias_filename = "bn_biases.npy"
    weights_filename = "bn_weights.npy"

    def __init__(self, layers):
        """ Initialization.

            layers:     List of layer neurons. [2 3 1]
                        [input <hidden_layers> output] -> [10 20 20 5]

            Initializes weights and biases randomly using Numpy's
            random array generator.
        """

        self.number_of_layers = len(layers)
        self.layers = layers
        self.weights = list()
        self.biases = list()

        self.biases = self.initialize_biases()
        self.weights = self.initialize_weights()
        print ">   Network initialized randomly."
        ws = list()
        t.flatten(self.weights, ws)
        t.plot_histogram(ws)

    def initialize_weights(self):
        """ Randomly initialize weights.
        """

        # WEIGHTS ---------------------------------------------------------------------------------
        # Similarly we create a transposed weights lists.
        # weights[0] -> weights connecting layers[0] -> layers[1]

        # Here we create a list of tuples then represent the weights connections.
        # Example:  layers = [A B C] where A, B, C are integers.
        #           zipped = [(A, B), (B, C)]
        zipped = zip(self.layers[:-1], self.layers[1:])

        return [np.random.randn(y, x) for x, y in zipped]
        # Example:  layers = [2, 3, 1]
        #           weights = [ array([[ 0.75054289, -0.18324735],
        #                               [ 0.32141292, -0.54226539],
        #                               [-0.53575605,  0.25761202]]),
        #                       array([[ 0.22304971,  1.29879581, -0.49437018]])]

    def initialize_biases(self):
        """ Randomly initialize biases.
        """

        # BIASES. ---------------------------------------------------------------------------------
        # Each neuron in each layer has its own bias, but the biases in
        # input layer (layers[0]) are ignored because, biases in general
        # are used for calculating the output of the neurons.

        # We basically create a vector via transpose... column vectors are need.
        # If a layer has 3 neurons, randn(3,1) creates a 3x1 matrix, aka a vector.
        # Which represented in Python in the following way:
        # [[0.1],
        #  [0.2],
        #  [0.1]]

        # biases[0] -> biases in layers[1]
        return [np.random.randn(x, 1) for x in self.layers[1:]]

    def save_state(self):
        """ Save the current state of the network.

            Might be useful for performance and time limitations reasons.

            This way, training can be completed in partial trainings.
        """

        np.save(self.state_folder + self.bias_filename, self.biases)
        np.save(self.state_folder + self.weights_filename, self.weights)
        print ">   Network state is saved."

    def load_state(self):
        """ Loads state from save *npy file.
        """

        # TODO: Check if state exists.
        b = np.load(self.state_folder + self.bias_filename)
        self.biases = b.tolist()

        w = np.load(self.state_folder + self.weights_filename)
        self.weights = w.tolist()
        print ">   Network state is loaded."

    def process(self, a):
        """ Input to the system is processed through the layers of the network,
            and an appropriate output is produced.

            a   Input to the system. Should be in the form of colunm vector.
                Just as np.random.randn(3,1) would produces a (3x1)
                column vector.
        """
        raise NotImplementedError("Process method must be implemented.")

    def train(self, training_data, epochs, eta, test_data=None):
        """ Training the neural network using training_data.

            training_data       : Training data. List of tuples (input, output)
            epochs              : Training epochs.
            eta                 : Learning rate.
            test_data           : If we want to test while training.
        """
        raise NotImplementedError("Training method must be implemented.")

    def cost_function_prime(self, output_activation, y):
        """ Derivation of the used cost function.
        """
        raise NotImplementedError("Cost function derivation must be implemented.")

    def evaluate(self, test_data):
        """ Evaluate the neural network success rate.

            Returns match rate as percentage.
        """
        results = [
            (np.argmax(self.process(x)), np.argmax(y)) for (x, y) in test_data
            ]

        matched = sum([1 for (x, y) in results if x == y])

        return 100.0 * matched / (len(test_data) * 1.0)

