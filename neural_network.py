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

        # We basically take each flow the information through the layers
        # until the output is reached.
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ A stochastic gradient descent method.

            training_data       : Training data. List of tuples (input, output)
            epochs              : Training epochs.
            mini_batch_size     : Sample size for stochastic gradient descent.
            eta                 : Learning rate.
            test_data           :

            Each epoch start with a randomized training data, then we divide it
            into mini batches. This gives a way to get easy random samples.

            And for each batch is we apply a gradient descent.
        """

        if test_data:
            n_test = len(test_data)

        n = len(training_data)                                      # Training data length.

        print ">   Epochs \t: ", epochs
        print ">   Eta \t:", eta
        print ""

        # xrange for lazy iteration generation based on epochs number.
        for e in xrange(epochs):
            np.random.shuffle(training_data)                        # Shuffling the data to increase the randomness of training.
            mini_batches = [training_data[k:k+mini_batch_size]      # Creating mini batches for stochastic method.
                            for k in xrange(0, n, mini_batch_size)]

            print ">   Epoch {0} : Updating weights and biases.".format(e)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # Evaluating current state with test data.
            if test_data:
                print ">   Epoch {0}: Completed.".format(e)
                print ">   Epoch {0}: Match rate: {1})".format(e, self.evaluate(test_data))
            else:
                print ">   Epoch {0}: Completed.".format(e)

    def update_mini_batch(self, batch, eta):
        """ Updates the network weights and biases by applying
            gradient descent on a single mini-batch

            batch  : List of (input, output) tuples.
            eta         : Learning rate.

            For every training example, gradients are calculated via
            backprop() method, then weights and biases are updated
            accordingly.
        """

        # creating matrices depending on bias and weight dimensions.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (eta / len(batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = t.sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            t.sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.number_of_layers):
            z = zs[-l]
            sp = t.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activation, y):
        return (output_activation - y)

    def evaluate(self, test_data):
        """ Return match rate as percentage.
        """
        results = [
            (np.argmax(self.feed_forward(x)), np.argmax(y)) for (x, y) in test_data
        ]

        matched = sum([1 for (x, y) in results if x == y])

        return 100.0 * matched / (len(test_data) * 1.0)

