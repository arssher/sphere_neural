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
        self.N = sizes[0]  # number of features
        self.K = sizes[-1]  # answer's dimension -- number of neurons in the last layer
        # L-1 column vectors containing biases, length of each equals to number of neurons:
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
        """Trains the network using stochastic gradient descendant.

        Args:
            train_data: set of training samples without answers, numpy ndarray of shape (M, N) where M is the number of
                samples and N is the number of features -- a usual sklearn input format.
            train_target: set of answers for train_data, numpy ndarray of shape (M, K) where M is the number of samples
                and K is the number of neurons in the last layer.
            epochs: number of epochs, i.e. iterations over the full training dataset.
            mini_batch_size: number of samples in one batch, i.e. after every mini_batch_size samples weights will be
                updated
            eta: learning rate
        """
        assert(train_data.shape[0] == train_target.shape[0])
        assert(train_data.shape[1] == self.N)
        assert(train_target.shape[1] == self.K)
        for ep_num in xrange(epochs):
            print "Starting epoch {0}".format(ep_num)
            train_data, train_target = shuffle_in_unison(train_data, train_target)
            for mini_batch_index in xrange(0, self.N, mini_batch_size):
                print "Starting minibatch {0}-{1}".format(mini_batch_index, mini_batch_index + mini_batch_size)
                mini_batch_train_data = train_data[mini_batch_index:mini_batch_index + mini_batch_size]
                mini_batch_train_target = train_target[mini_batch_index:mini_batch_index + mini_batch_size]
                self.update_mini_batch(mini_batch_train_data, mini_batch_train_target, eta)

            print "Epoch {0} complete".format(ep_num)

    def update_mini_batch(self, train_data, train_target, eta):
        """Calculate weight gradients for each neuron in each layer and update weights using train_data-train_target
           samples.

        Args:
            train_data: set of training samples without answers, numpy ndarray of shape (M, N) where M is the number of
                samples and N is the number of features -- a usual sklearn input format. However, we will immediately
                transpose them (and train_target too) since it is more convenient for the math.
            train_target: set of answers for train_data, numpy ndarray of shape (M, K) where M is the number of samples
                and K is the number of neurons in the last layer.
            eta: learning rate
        """
        assert(train_data.shape[0] == train_target.shape[0])
        assert (train_data.shape[1] == self.N)
        assert(train_target.shape[1] == self.K)
        # TODO: rewrite using matrix math
        train_data_transposed = np.transpose(train_data)
        train_target_transposed = np.transpose(train_target)
        # accumulates changes in biases per whole batch
        nabla_b = [np.zeros(layer_biases.shape) for layer_biases in self.biases]
        # accumulates changes in weights per whole batch
        nabla_w = [np.zeros(layer_weights.shape) for layer_weights in self.weights]
        for i_sample in xrange(train_data.shape[0]):
            x_col = train_data_transposed[:, i_sample].reshape((self.N, 1))
            y_col = train_target_transposed[:, i_sample].reshape((self.K, 1))
            delta_nabla_b, delta_nabla_w = self.backprop(x_col, y_col)
            nabla_b = map(lambda nb, dnb: nb + dnb, nabla_b, delta_nabla_b)
            nabla_w = map(lambda nw, dnw: nw + dnw, nabla_w, delta_nabla_w)

        self.weights = [layer_weights - (eta / train_data.shape[0]) * nw
                        for layer_weights, nw in zip(self.weights, nabla_w)]
        self.biases = [layer_biases - (eta / train_data.shape[0]) * nb
                       for layer_biases, nb in zip(self.biases, nabla_b)]

    # returns tuple (nabla_b, nabla_w), shape correspond to those of self.biases and self.weights
    def backprop(self, x, y):
        """Calculate weight gradient for each neuron in each layer using single sample x with answer y.

        Args:
            x: Single sample to train on, numpy ndarray. x has shape (N, 1) where N is the number of features.
            y: Answer to x. y has shape (K, 1) where K is the number of neurons in the last layer.

        Returns:
            A tuple (delta_nabla_b, delta_nabla_w) where delta_nabla_b is the list of biases's gradients changes, and
             delta_nabla_w is the list of weights's gradients changes. Their shape is exactly like self.biases and
             self.weights -- list of numpy column vectors and list of numpy matrices.
        """
        assert(x.shape == (self.N, 1))
        assert(y.shape == (self.K, 1))
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forward
        activation = x
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