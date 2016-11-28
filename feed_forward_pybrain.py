"""
    feed_forward_pybrain.py
    ~~~~~~~~~~~~~~~

    A feed forward neural network using PyBrain library.
"""

import minst_data as minst
import numpy as np

# Pybrain imports
from pybrain.datasets import ClassificationDataSet, SupervisedDataSet
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer, TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.customxml import NetworkReader, NetworkWriter

class FeedForwardPB(object):
    # For saving network state.
    network_fn = "feed_forward_pb.xml"

    # Data path.
    data_path = "data/mnist_data.pkl.gz"

    def __init__(self, input_layer=784, hidden_layer=30, output_layer=10, load=False):
        """ Initiate the network. """

        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

        # Loading from a previously saved state.
        if load:
            self.net = NetworkReader.readFrom(self.network_fn)
        else:
            self.net = buildNetwork(input_layer, hidden_layer, output_layer, bias=True, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer)

        # Creating training and testing data.
        print ">   Reading data."
        data = minst.MnistData(self.data_path)

        # Reading the data.
        trd = data.get_training_data()
        ttd = data.get_test_data()
        vad = data.get_validation_data()

        # Converting into shape that can be used by the library.
        print ">   Creating training dataset."
        self.ds_training = SupervisedDataSet(input_layer, output_layer)

        for x, y in trd:
            self.ds_training.addSample(x.reshape(input_layer), y.reshape(output_layer))

        print ">   Creating testing dataset."
        self.ds_test = SupervisedDataSet(input_layer, output_layer)
        for x, y in ttd:
            self.ds_test.addSample(x.reshape(input_layer), y.reshape(output_layer))

        print ">   Creating validation dataset."
        self.ds_validation = SupervisedDataSet(input_layer, output_layer)
        for x, y in vad:
            self.ds_validation.addSample(x.reshape(input_layer), y.reshape(output_layer))

        print ">   All data loaded."

        # Default trainer for the network.
        print ">   Creating default trainer. (Use update_learningrate if you want to change)"
        self.trainer = BackpropTrainer(self.net, self.ds_training, learningrate=1)

    def update_learningrate(self, val):
        """ Update the trainer with the current learning rate. """

        print ">   Training rate is update to: {0}".format(val)
        self.trainer = BackpropTrainer(self.net, self.ds_training, learningrate=val)

    def evaluate(self, validation=False):
        """ Evaluate network using test data or validation data."""

        if validation:
            data = zip(self.ds_validation["input"], self.ds_validation["target"])
        else:
            data = zip(self.ds_test["input"], self.ds_test["target"])

        results = [
            (self.activate(x, True), np.argmax(y)) for (x, y) in data
            ]

        matched = sum([1 for (x, y) in results if x == y])

        return 100.0 * matched / (len(data) * 1.0)

    def save_nn_state(self):
        """ Save neural network state so it can be loaded later on."""

        NetworkWriter.writeToFile(self.net, self.network_fn)

    def activate(self, inp, number=False):
        """ Get the output for the given input. """

        result = self.net.activate(inp)

        if number:
            return np.argmax(result)
        else:
            return result

    def train(self, epochs=1, save=True):
        """ Train the network certain number of times. """

        error = self.evaluate()
        errors = [error]

        print ">   Initial match: {0}%".format(error)

        for e in range(0, epochs):
            print ">   Training: epoch {0}".format(e+1)
            self.trainer.train()
            error = self.evaluate()
            print ">           : match {0}%".format(error)
            errors.append(error)

            if save:
                self.save_nn_state()

