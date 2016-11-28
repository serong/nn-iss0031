"""
    train_keras.py
    ~~~~~~~~~~~~~~

    Training a neural network using Keras library and TensorFlow backend.
"""

from feed_forward_keras import FeedForwardKeras

# Creating a default network.
nn = FeedForwardKeras()

# Training for 5 epochs.
nn.train(5, verbose=1)

# Evaluating the result.
nn.evaluate()