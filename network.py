import numpy as np
from utils import shuffle_in_unison

# L is number of layers
# While working with a (l-1), l pair of layers, let K be the number of neurons in (l-1) layer and J number of neurons
# in l layer. weights[l][j, k] contains a weight connecting k'th neuron from layer l-1 to j'th neuron in layer l.
# We will count layers from 1, so the input layer is number 0 layer.

class Network(object):
    # sizes is a list of layer's sizes
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # L-1 column vectors containing biases, length of each equals to number of layers:
        # if layer has J neurons, it's bias has Jx1 shape
        # We don't need biases for the first (input) layer, bias[0] is the layer 1's biases.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # L-1 weight matrices. Matrix weights[0] connects input layer 0 with layer 1.
        # The matrix weights[l] has size JxK, where
        # K is number of neurons in the layer l (the previous one), and J is number of neurons in the layer l+1.
        self.weights = [np.random.randn(J, K)
                        for K, J in zip(sizes[:-1], sizes[1:])]

    # network's output for input vector a, a must have shape sizes[0]x1 or (sizes[0], ) -- the latter will be reshaped
    def feedforward(self, a):
        assert(a.shape == (self.sizes[0], ) or a.shape == (self.sizes[0], 1))
        a = a.reshape((self.sizes[0], 1))
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    # evaluate test data and print percent of correct answers
    def evaluate_mnist(self, test_data, test_target):
        correct_answers = 0
        for test_x, test_y in zip(test_data, test_target):
            answer = np.argmax(self.feedforward(test_x))
            # print "given answer is {0}, right answer is {1}".format(answer, test_y)
            if answer == test_y:
                correct_answers += 1
        print "Evaluation finished: {0} / {1} tests correct".format(correct_answers, len(test_data))
        return correct_answers

    def evaluate_xor(self, test_data, test_target):
        correct_answers = 0
        for test_x, test_y in zip(test_data, test_target):
            network_answer = self.feedforward(test_x)
            if round(network_answer[0][0]) == test_y[0]:
                correct_answers += 1
            print "test is {0}. network answer was {1}, correct answer is {2}".format(test_x, network_answer, test_y)
        print "xor evaluation finished, {0} of {1} answers correct".format(correct_answers, test_data.shape[0])

    def SGD(self, train_data, train_target, epochs, mini_batch_size, eta):
        assert(train_data.shape[1] == self.sizes[0])
        N = len(train_data)
        for j in xrange(epochs):
            print "Starting epoch {0}".format(j)
            # train_data, train_target = shuffle_in_unison(train_data, train_target)
            for mini_batch_index in xrange(0, N, mini_batch_size):
                print "Starting minibatch {0}-{1}".format(mini_batch_index, mini_batch_index + mini_batch_size)
                mini_batch_train_data = train_data[mini_batch_index:mini_batch_index + mini_batch_size]
                mini_batch_train_target = train_target[mini_batch_index:mini_batch_index + mini_batch_size]
                self.update_mini_batch(mini_batch_train_data, mini_batch_train_target, eta)

            print "Epoch {0} complete".format(j)

    def update_mini_batch(self, train_data, train_target, eta):

        nabla_b = [np.zeros(layer_biases.shape) for layer_biases in self.biases]  # accumulates changes in biases per whole batch
        nabla_w = [np.zeros(layer_weights.shape) for layer_weights in self.weights]  # accumulates changes in weights per whole batch
        for x, y in zip(train_data, train_target):
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = map(lambda nb, dnb: nb + dnb, nabla_b, delta_nabla_b)
            nabla_w = map(lambda nw, dnw: nw + dnw, nabla_w, delta_nabla_w)

        self.weights = [layer_weights - (eta / train_data.shape[0]) * nw for layer_weights, nw in zip(self.weights, nabla_w)]
        self.biases = [layer_biases - (eta / train_data.shape[0]) * nb for layer_biases, nb in zip(self.biases, nabla_b)]

    # returns tuple (nabla_b, nabla_w), shape correspond to those of self.biases and self.weights
    def backprop(self, x, y):
        y = y.reshape((self.sizes[-1], 1))
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forward
        activation = x.reshape((self.sizes[0], 1))
        activations = [activation]  # list to store all the activation vectors, layer by layer
        zs = []  # list to store z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward
        # activaitons[-1] is the final output
        assert(zs[-1].shape == y.shape) # FIXME: type checking?
        delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        cost(activations, y)
        return nabla_b, nabla_w

# returns a column vector -- derivative of C along all final activations
def cost_derivative(output_activations, y):
    return output_activations - y

def cost(activations, target):
    pass
    # print target

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_hack(z):
    if z > 0:
        e = np.exp(-z)
        assert np.isfinite(e)
        return 1.0 / (1 + e)
    else:
        e = np.exp(z)
        assert np.isfinite(e)
        return e / (1 + e)

sigmoid_hack_vectorizer = np.vectorize(sigmoid_hack)


# Derivative of the sigmoid function
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))