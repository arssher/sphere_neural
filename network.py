import numpy as np


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
            a = sigmoid_hack_vectorizer(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data, test_target):
        correct_answers = 0
        for test_x, test_y in zip(test_data, test_target):
            answer = np.argmax(self.feedforward(test_x))
            if answer == test_y:
                correct_answers += 1

        print "Evaluation finished: {0} / {1} tests correct".format(correct_answers, len(test_data))
        return correct_answers


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