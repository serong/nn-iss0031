import neural_network as network
import tools as tools
import minst_data as minst
from feed_forward import FeedForward


PATH = "data/mnist_data.pkl.gz"
print ">>> Unpacking data."
data = minst.MnistData(PATH)

training_data = data.get_training_data()
test_data = data.get_test_data()
validation_data = data.get_validation_data()

# net = network.NeuralNetwork([784, 30, 10])
#
# number3 = validation_data[0]
#
# net.SGD(training_data, 10, 10, 3.0, test_data)


net = FeedForward([784, 30, 10])
net.load_state()
net.train(training_data, 10, 3.0, test_data)



