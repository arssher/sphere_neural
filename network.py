import numpy as np
from utils import shuffle_in_unison

# force numpy to throw exception on any math errors
np.seterr(all='raise')

# L is number of layers
# While working with a (l-1), l pair of layers, let K be the number of neurons in (l-1) layer and J number of neurons
# in l layer. weights[l][j, k] contains a weight connecting k'th neuron from layer l-1 to j'th neuron in layer l.
# We will count layers from 1, so the input layer is number 0 layer.


class Network(object):
    # sizes is a list of layer's sizes
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        assert(self.num_layers >= 2)
        self.sizes = sizes
        self.N = sizes[0]  # number of features
        self.K = sizes[-1]  # answer's dimension -- number of neurons in the last layer
        self.biases = None
        self.weights = None
        self.init_weights_and_biases()

    def init_weights_and_biases(self):
        # L-1 column vectors containing biases, length of each equals to number of neurons:
        # if layer has J neurons, it's bias has Jx1 shape
        # We don't need biases for the first (input) layer, bias[0] is the layer 1's biases.
        mu, sigma = 0, 1.0
        self.biases = [np.random.normal(mu, sigma, (y, 1)) for y in self.sizes[1:]]
        # L-1 weight matrices. Matrix weights[0] connects input layer 0 with layer 1.
        # The matrix weights[l] has size JxK, where
        # K is number of neurons in the layer l (the previous one), and J is number of neurons in the layer l+1.
        self.weights = [np.random.normal(mu, sigma, (J, K))
                        for K, J in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, x):
        """Run sample a through the trained network and return final activations.

           Args:
               x: sample to run. Numpy ndarray of shape (self.N, 1) or (self.N, ). The latter will be reshaped.

            Returns:
                Tuple (ouput, activations, zs), where:
                  output is the final activation, i.e. output of the network. Numpy ndarray of shape (self.K, 1).
                  activations is a (python) list of activations for each layer. Each element is ndarray with
                   shape (J, 1) where J is the number of neurons in the corresponding layer.
                  zs is a (python) list of z for each layer. Each element is ndarray with
                   shape (J, 1) where J is the number of neurons in the corresponding layer. z of a neuron is a weighted
                   sum of the neuron's inputs: activation_func(z) === activation.
        """
        assert(x.shape == (self.sizes[0],) or x.shape == (self.sizes[0], 1))
        x = x.reshape((self.sizes[0], 1))
        activation = x
        activations = [activation] # list to store all the activation vectors, layer by layer
        zs = [] # list to store z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        return activations[-1], activations, zs

        # # forward
        # activation = x
        # activations = [activation]  # list to store all the activation vectors, layer by layer
        # zs = []  # list to store z vectors, layer by layer
        # for b, w in zip(self.biases, self.weights):
        #     z = np.dot(w, activation) + b
        #     zs.append(z)
        #     activation = sigmoid(z)
        #     activations.append(activation)

    # evaluate test data and print percent of correct answers
    def evaluate_mnist(self, test_data, test_target):
        correct_answers = 0
        for test_x, test_y in zip(test_data, test_target):
            answer = np.argmax(self.feedforward(test_x)[0])
            # print "given answer is {0}, right answer is {1}".format(answer, test_y)
            if answer == test_y:
                correct_answers += 1
        print "Evaluation finished: {0} / {1} tests correct".format(correct_answers, len(test_data))
        return correct_answers

    def evaluate_xor(self, test_data, test_target):
        correct_answers = 0
        for test_x, test_y in zip(test_data, test_target):
            network_answer = self.feedforward(test_x)[0]
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
        self.init_weights_and_biases()
        print "learning on {0} samples with {1} features".format(train_data.shape[0], self.N)
        for ep_num in xrange(epochs):
            print "Starting epoch {0}".format(ep_num)
            train_data, train_target = shuffle_in_unison(train_data, train_target)
            for mini_batch_index in xrange(0, train_data.shape[0], mini_batch_size):
                # print "Starting minibatch {0}-{1}".format(mini_batch_index, mini_batch_index + mini_batch_size)
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
        assert(train_data.shape[1] == self.N)
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
            nabla_b_for_i_sample, nabla_w_for_i_sample = self.backprop_single_sample(x_col, y_col)
            nabla_b = map(lambda nb, dnb: nb + dnb, nabla_b, nabla_b_for_i_sample)
            nabla_w = map(lambda nw, dnw: nw + dnw, nabla_w, nabla_w_for_i_sample)
            if False:
                for i in xrange(len(nabla_w)):
                    print "weights grads from layer {0} to layer {1}:".format(i, i+1)
                    print nabla_w_for_i_sample[i]

        self.weights = [layer_weights - (eta / train_data.shape[0]) * nw
                        for layer_weights, nw in zip(self.weights, nabla_w)]
        self.biases = [layer_biases - (eta / train_data.shape[0]) * nb
                       for layer_biases, nb in zip(self.biases, nabla_b)]

    # returns tuple (nabla_b, nabla_w), shape correspond to those of self.biases and self.weights
    def backprop_single_sample(self, x, y):
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
        output, activations, zs = self.feedforward(x)

        # backward
        # activaitons[-1] is the final output
        assert(zs[-1].shape == y.shape)
        delta = cost_derivative(output, y) * sigmoid_prime(zs[-1])
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