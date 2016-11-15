import neural_network as network
import tools as tools
import minst_data as minst


PATH = "data/mnist_data.pkl.gz"
print ">>> Unpacking data."
data = minst.MnistData(PATH)

# List of (x, y) tuples.
training_data = zip(data.get_training_images(), data.get_training_labels(True))
test_data = zip(data.get_test_images(), data.get_test_labels(True))
validation_data = zip(data.get_validation_images(), data.get_validation_labels(True))

net = network.NeuralNetwork([784, 30, 10])

print ">>> Initial Validation: "
number = validation_data[0]
print "\t\t Number: 3",
print "\t\t ", net.feed_forward(number[0])



