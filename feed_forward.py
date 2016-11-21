"""
    feed_forward.py
    ~~~~~~~~~~~~~~~

    A feed forward neural network using stochastic gradient descent
    and backpropogation.
"""

from base_network import BaseNetwork
import tools as t
import numpy as np

class FeedForward(BaseNetwork):
    weights_filename = "ff_weights.npy"
    bias_filename = "ff_biases.npy"

    # To be used with SGD training method.
    # Sample size for stochastic gradient descent.
    mini_batch_size = 10

    def __init__(self, layers):
        super(FeedForward, self).__init__(layers)

    def process(self, a):
        """ Feed forward processing of inputs.

            Input to the system is processed through the layers of the network,
            and an appropriate output is produced.

            a   Input to the system. Should be in the form of colunm vector.
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

    def train(self, training_data, epochs, eta, test_data=None):
        """ A stochastic gradient descent training method.

            training_data       : Training data. List of tuples (input, output)
            epochs              : Training epochs.
            eta                 : Learning rate.
            test_data           :

            Each epoch start with a randomized training data, then we divide it
            into mini batches. This gives a way to get easy random samples.

            And for each batch is we apply a gradient descent.
        """

        print ">   Starting the training."

        if test_data:
            n_test = len(test_data)

        # Training data len.
        n = len(training_data)

        print ">   Epochs \t: ", epochs
        print ">   Eta \t:", eta
        print ""

        print ">   Initial match rate: {0}%".format(self.evaluate(test_data))

        # xrange for lazy iteration generation based on epochs number.
        for e in xrange(epochs):

            # Shuffling the data to increase the randomness of training.
            np.random.shuffle(training_data)

            # Creating mini batches for stochastic method.
            # This divides the training data to mini batches with the size of
            # mini_batch_size.
            # If we have td = [0, 1, ..., 99] n = 100, mini_batch_size = 5
            # it becomes:
            #   [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], ..., [95, 96, 97, 98, 99]]
            mini_batches = [training_data[k:k+self.mini_batch_size]
                            for k in xrange(0, n, self.mini_batch_size)]

            print ">   Epoch {0} : Updating weights and biases.".format(e)
            for mini_batch in mini_batches:
                # Weights and biases are updated according to the gradient descent calculated
                # on each mini batch. Since we already randomized the training data before,
                # the mini batches are actually random. So, we're making random picks.
                self.__update_mini_batch(mini_batch, eta)

            # Evaluating current state with test data.
            if test_data:
                print ">   Epoch {0}: Completed.".format(e)
                print ">   Epoch {0}: Match rate: {1}%".format(e, self.evaluate(test_data))
            else:
                print ">   Epoch {0}: Completed.".format(e)

    def __update_mini_batch(self, batch, eta):
        """ Updates the network weights and biases by applying
            gradient descent on a single mini-batch

            batch  : List of (input, output) tuples.
            eta    : Learning rate.

            For every training example, gradients are calculated via
            backpropogation method, then weights and biases are updated
            accordingly.
        """

        # creating empty matrices depending on bias and weight dimensions.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Each batch is the size of self.mini_batch_size and contains tuples
        # of (image, label) or in this case (x, y)
        for x, y in batch:
            # Calculation of step change for supplied input and output... hence the error.
            delta_nabla_b, delta_nabla_w = self.__backpropogation(x, y)

            # Updating temporary matrices with delta values (gradient steps)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Each weight is updated, using learning rate and the calculated delta weights or biases.
        self.weights = [w - (eta / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def __backpropogation(self, x, y):
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
        delta = self.cost_function_prime(activations[-1], y) * \
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

    def cost_function_prime(self, output_activation, y):
        """ Derivation of the used cost function.
        """
        return (output_activation - y)
