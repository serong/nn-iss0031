from feed_forward_keras import FeedForwardKeras

nn = FeedForwardKeras()

nn.train(5, verbose=1)
nn.evaluate()