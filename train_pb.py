"""
    train_pb.py
    ~~~~~~~~~~~

    Training a feed forward neural network using PyBrain.
"""
from feed_forward_pybrain import FeedForwardPB

# Creating the neural network with default values.
# [784, 30, 10] as layers.
nn = FeedForwardPB()
nn.update_learningrate(0.5)

# Training for 5 times.
nn.train(5)