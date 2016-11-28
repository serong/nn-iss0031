"""
    train.py
    ~~~~~~~~

    Training a neural network based on numpy implementation. More information
    can be found in FeedForward class located in feed_forward.py file.
"""

import minst_data as minst
from feed_forward import FeedForward

# Loading the data from mnist file.
# TODO: Include in the class?
PATH = "data/mnist_data.pkl.gz"
data = minst.MnistData(PATH)

print ">>> Unpacking data."

# Converting data into a shape that can be used by the network.
training_data = data.get_training_data()
test_data = data.get_test_data()
validation_data = data.get_validation_data()

# Creating standard neural network.
net = FeedForward([784, 30, 10])

# Load a previously trained network.
# This currently just loads the weights and biases without actually checking
# if they are the right weights.
# net.load_state()

# Training for 10 epochs with 3.0 learning rate.
net.train(training_data, 10, 3.0, test_data)

# Saving the state so it can be used later on.
net.save_state()


