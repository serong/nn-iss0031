"""
    minst_data
    ~~~~~~~~~~

    Loading MNIST data.
"""

import pickle, gzip
import numpy as np
PATH = "data/mnist_data.pkl.gz"

class MnistData():
    def __init__(self, path):
        """ Load the raw data from a pickle file.
            Pickle file has 3 tuples of (image, label) which are:
                training_date, validation_data, test_data.

            Initially:
            data => ([784-d vectors], [int])
        """
        file = gzip.open(path, "rb")

        # Pickle file loads 3 seperate tuples... each a (image, label) couple.
        self.training_data, self.validation_data, self.test_data = pickle.load(file)
        file.close()

    def make_vector(self, num):
        """ Create 10-d vector out of given num.

            So, 1 => [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]T
        """
        zeros = np.zeros((10, 1))       #
        zeros[num] = 1.0
        return zeros

    def get_training_images(self):
        return self.training_data[0]

    def get_validation_images(self):
        return self.validation_data[0]

    def get_test_images(self):
        return self.test_data[0]

    def get_training_labels(self, vector=False):
        labels = self.training_data[1]
        if vector:
            labels_v = list()
            for x in labels:
                labels_v.append(self.make_vector(x))
            return labels_v
        else:
            return labels

    def get_validation_labels(self, vector=False):
        labels = self.validation_data[1]
        if vector:
            labels_v = list()
            for x in labels:
                labels_v.append(self.make_vector(x))
            return labels_v
        else:
            return labels

    def get_test_labels(self, vector=False):
        labels = self.test_data[1]
        if vector:
            labels_v = list()
            for x in labels:
                labels_v.append(self.make_vector(x))
            return labels_v
        else:
            return labels
