import tools as tools
import numpy as np
import minst_data as minst
from keras.models import Sequential
from keras.layers import Dense, Activation

class FeedForwardKeras(object):
    data_path = "data/mnist_data.pkl.gz"

    def __init__(self, input_layer=784, hidden_layer=30, output_layer=10, load=False):
        """ Create a default model. """

        print ">>> Preparing data."
        self.training_x = np.random.random((50000, 784))
        self.test_x = np.random.random((10000, 784))
        self.validation_x = np.random.random((10000, 784))

        self.training_y = np.random.randint(2, size=(50000, 10))
        self.test_y = np.random.randint(2, size=(10000, 10))
        self.validation_y = np.random.randint(2, size=(10000, 10))

        self.prepare_data()

        print ">   Creating the model."
        self.net = Sequential()

        # Adding the layers.
        self.net.add(Dense(30, input_dim=784, activation="sigmoid"))
        self.net.add(Dense(10, activation="softmax"))

        # Compiling the network.
        # optimizers    : sgd, rmsprop, adagrad, adadelta, adam, adamax, tfoptimizer
        #                 https://keras.io/optimizers/
        # loss          : mean_squared_error, etc.
        #                 https://keras.io/objectives/
        print ">   Compiling the model. (sgd, mse)"
        self.net.compile(optimizer="rmsprop",
                         loss="mse",
                         metrics=["accuracy"])

    def prepare_data(self):
        """ Preparing data. """
        data = minst.MnistData(self.data_path)

        training_data = data.get_training_data()
        test_data = data.get_test_data()
        validation_data = data.get_validation_data()

        print ">   Formatting training data."

        i = 0
        while i < len(training_data):
            self.training_x[i] = training_data[i][0].reshape(784)
            self.training_y[i] = training_data[i][1].reshape(10)
            i += 1

        print ">   Formatting testing and validation data."

        i = 0
        while i < len(test_data):
            self.test_x[i] = test_data[i][0].reshape(784)
            self.validation_x[i] = validation_data[i][0].reshape(784)

            self.test_y[i] = test_data[i][1].reshape(10)
            self.validation_y[i] = validation_data[i][1].reshape(10)
            i = i + 1

    def train(self, epochs, batch_size = 30, verbose=0):
        """ Train the model. """
        print ">   Training the model for {0} epochs.".format(epochs)

        self.net.fit(self.training_x,
                     self.training_y,
                     nb_epoch=epochs,
                     verbose=verbose)

    def evaluate(self):
        """ Evaluate the matches. """
        res = self.net.evaluate(self.test_x, self.test_y, verbose=0)

        print ">   Match: {0}%".format(res[1] * 100)

