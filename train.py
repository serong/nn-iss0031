import tools as tools
import minst_data as minst
from feed_forward import FeedForward


PATH = "data/mnist_data.pkl.gz"
print ">>> Unpacking data."
data = minst.MnistData(PATH)

training_data = data.get_training_data()
test_data = data.get_test_data()
validation_data = data.get_validation_data()

# Creating standard neural network.
net = FeedForward([784, 30, 10])
# net.load_state()

# Training for 10 epochs with 3.0 learning rate.
net.train(training_data, 10, 3.0, test_data)

# Drawing histograms of weights and biases.
# ws = list()
# bs = list()
#
# tools.flatten(net.weights, ws)
# tools.flatten(net.biases, bs)
# tools.plot_histogram(ws)
# tools.plot_histogram(bs)



