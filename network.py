import numpy as np
from utils import shuffle_in_unison

# force numpy to throw exception on any math errors
np.seterr(all='raise')

# Some used names:
# L is number of layers
# While working with a (l-1), l pair of layers, let K be the number of neurons in (l-1) layer and J number of neurons
# in l layer. weights[l][j, k] contains a weight connecting k'th neuron from layer l-1 to j'th neuron in layer l.
# We will count layers from 0, so the input layer is number 0 layer.
# z of a neuron is a weighted sum of the neuron's inputs: activation_func(z) === activation
# M is usually number of samples


class Network(object):
    # sizes is a list of layer's sizes
    def __init__(self, sizes, activation, cost, gradient_check):
        assert sizes is not None
        self.num_layers = len(sizes)
        assert(self.num_layers >= 2)
        self.sizes = sizes
        self.N = sizes[0]  # number of features
        self.P = sizes[-1]  # answer's dimension -- number of neurons in the last layer
        self.biases = None
        self.weights = None
        self.init_weights_and_biases()

        self.activation_func = sigmoid
        self.activation_func_derivative = sigmoid_derivative
        if activation == "relu":
            self.activation_func = relu
            self.activation_func_derivative = relu_derivative
        elif activation == "identity":
            self.activation_func = identity_activation
            self.activation_func_derivative = identity_activation_derivative
        self.cost_func = msi
        self.cost_derivative = msi_derivative
        if cost == "crossentropy":
            pass

        self.gradient_check = gradient_check

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

    def feedforward(self, X, weights=None, biases=None):
        """Run samples X through the trained network and return outputs, activations and weighted sums for each sample

           Args:
               X: samples to run. Numpy ndarray of shape (self.N, M) or (self.N, ), where M is number of samples.
                The latter (one sample) will be reshaped.
               weights: you can specify custom weights here, they will be used instead of self.weights. Useful for
                 debugging (gradient-checking, you know).
               biases: you can specify custom biases here, they will be used instead of self.biases. Useful for
                 debugging (gradient-checking, you know).

            Returns:
                Tuple (ouputs, activations, zs), where:
                  output is the final activation, i.e. output of the network, for each sample.
                   Numpy ndarray of shape (self.P, M).
                  activations is a (python) list of activations for each layer for each sample (that's why double s).
                   Each element is ndarray with shape (J, M) where J is the number of neurons in the corresponding
                    layer.
                  zs is a (python) list of z for each layer for each sample. Each element is ndarray of
                   shape (J, M) where J is the number of neurons in the corresponding layer.
        """
        if X.shape == (self.N,):
            X = X.reshape((self.N, 1))
        assert (X.shape[0] == self.N)

        if weights is None:
            weights = self.weights
        if biases is None:
            biases = self.biases

        activation = X
        activations = [activation]  # list to store all the activation vectors, layer by layer
        zs = []  # list to store z vectors, layer by layer
        for b, w in zip(biases, weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_func(z)
            activations.append(activation)

        return activations[-1], activations, zs

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
            train_target: set of answers for train_data, numpy ndarray of shape (M, P) where M is the number of samples
                and P is the number of neurons in the last layer.
            epochs: number of epochs, i.e. iterations over the full training dataset.
            mini_batch_size: number of samples in one batch, i.e. after every mini_batch_size samples weights will be
                updated
            eta: learning rate
        """
        assert(train_data.shape[0] == train_target.shape[0])
        assert(train_data.shape[1] == self.N)
        assert(train_target.shape[1] == self.P)
        self.init_weights_and_biases()
        print "learning on {0} samples with {1} features, minibatch size is {2}"\
            .format(train_data.shape[0], self.N, mini_batch_size)
        for ep_num in xrange(epochs):
            print "Starting epoch {0}".format(ep_num)
            train_data, train_target = shuffle_in_unison(train_data, train_target)
            for mini_batch_index in xrange(0, train_data.shape[0], mini_batch_size):
                # print "Starting minibatch {0}-{1}".format(mini_batch_index, mini_batch_index + mini_batch_size)
                mini_batch_train_data = train_data[mini_batch_index:mini_batch_index + mini_batch_size]
                mini_batch_train_target = train_target[mini_batch_index:mini_batch_index + mini_batch_size]
                dLdW, dLdB = self.update_mini_batch_one_by_one(mini_batch_train_data, mini_batch_train_target, eta)

            print "Epoch {0} complete".format(ep_num)

    def update_mini_batch_one_by_one(self, train_data, train_target, eta):
        """Calculate weight and bias derivatives for each neuron in each layer and update self.weights & self.biases
           using train_data-train_target samples. It means that for each sample in train_data we calculate the
           derivatives, sum them up and update self.weights and self.biases using that summed value.

        Args:
            train_data: set of training samples without answers, numpy ndarray of shape (M, N) where M is the number of
                samples and N is the number of features -- a usual sklearn input format. However, we will immediately
                transpose them (and train_target too) since it is more convenient for the math.
            train_target: set of answers for train_data, numpy ndarray of shape (M, K) where M is the number of samples
                and K is the number of neurons in the last layer.
            eta: learning rate

        Returns:
            Tuple (nabla_w, nabla_b) with accumulated weights and biases changes, shape like self.weights and
              self.biases exactly. It is useful only for debugging, the main action happens here as a side effect --
              function changes self.weights & self.biases itself.
        """
        assert(train_data.shape[0] == train_target.shape[0])
        assert(train_data.shape[1] == self.N)
        assert(train_target.shape[1] == self.P)
        # TODO: rewrite using matrix math
        train_data_transposed = np.transpose(train_data)
        train_target_transposed = np.transpose(train_target)
        # accumulates changes in biases per whole batch
        nabla_b = [np.zeros(layer_biases.shape) for layer_biases in self.biases]
        # accumulates changes in weights per whole batch
        nabla_w = [np.zeros(layer_weights.shape) for layer_weights in self.weights]
        for i_sample in xrange(train_data.shape[0]):
            x_col = train_data_transposed[:, i_sample].reshape((self.N, 1))
            y_col = train_target_transposed[:, i_sample].reshape((self.P, 1))
            nabla_b_for_i_sample, nabla_w_for_i_sample = self.backprop(x_col, y_col)
            nabla_b = map(lambda nb, dnb: nb + dnb, nabla_b, nabla_b_for_i_sample)
            nabla_w = map(lambda nw, dnw: nw + dnw, nabla_w, nabla_w_for_i_sample)

        # WARNING: For performance reasons, I decided not to make copies of self.weights while calculating
        # derivatives manually, I modify them in place and the restore old value.
        # Therefore, in case of mistakes in gradient checking this may lead to very nasty bugs.
        # Be careful and disable it if something goes wrong.
        if self.gradient_check:
            for layer_i in xrange(len(self.sizes) - 1):
                self.grad_check_per_layer_per_minibatch(train_data_transposed,
                                                        train_target_transposed,
                                                        layer_i, nabla_w[layer_i])

        self.weights = [layer_weights - (eta / train_data.shape[0]) * nw
                        for layer_weights, nw in zip(self.weights, nabla_w)]
        self.biases = [layer_biases - (eta / train_data.shape[0]) * nb
                       for layer_biases, nb in zip(self.biases, nabla_b)]
        return nabla_w, nabla_b

    def update_mini_batch(self, train_data, train_target, eta):
        """Calculate weight and bias derivatives for each neuron in each layer and update self.weights & self.biases
        using train_data-train_target samples. It means that for each sample in train_data we calculate the
        derivatives, sum them up and update self.weights and self.biases using that summed value.

        Args:
            train_data: set of training samples without answers, numpy ndarray of shape (M, N) where M is the number of
                samples and N is the number of features -- a usual sklearn input format. However, we will immediately
                transpose them (and train_target too) since it is more convenient for the math.
            train_target: set of answers for train_data, numpy ndarray of shape (M, P) where M is the number of samples
                and P is the number of neurons in the last layer.
            eta: learning rate

        Returns:
            Tuple (dLdW, dLdB) with accumulated weights and biases changes, shape like self.weights and
              self.biases exactly. It is useful only for debugging, the main action happens here as a side effect --
              function changes self.weights & self.biases itself.
        """
        assert(train_data.shape[0] == train_target.shape[0])
        assert(train_data.shape[1] == self.N)
        assert(train_target.shape[1] == self.P)
        train_data_transposed = np.transpose(train_data)
        train_target_transposed = np.transpose(train_target)

        dLdB, dLdW = self.backprop(train_data_transposed, train_target_transposed)

        self.biases = [layer_biases - (eta / train_data.shape[0]) * nb
                       for layer_biases, nb in zip(self.biases, dLdB)]
        self.weights = [layer_weights - (eta / train_data.shape[0]) * nw
                        for layer_weights, nw in zip(self.weights, dLdW)]
        return dLdW, dLdB

    def backprop_single_sample(self, x, y):
        """Calculate weight and bias derivatives for each neuron in each layer using single sample x with answer y via
        backpropagation.

        Args:
            x: Single sample to train on, numpy ndarray. x has shape (N, 1) where N is the number of features.
            y: Answer to x. y has shape (P, 1) where P is the number of neurons in the last layer.

        Returns:
            A tuple (dLdB, dLdW) where dlDb contains derivatives of L (loss or cost function) along each bias and
            dLdW contains derivatives of L along each weight. Their shape is exactly the same as self.biases and
            self.weights -- list of numpy column vectors and list of numpy matrices.
        """
        assert(x.shape == (self.N, 1))
        assert(y.shape == (self.P, 1))
        dLdB = [np.zeros(b.shape) for b in self.biases]
        dLdW = [np.zeros(w.shape) for w in self.weights]

        # forward
        output, activations, zs = self.feedforward(x)

        # backward
        assert(zs[-1].shape == y.shape)
        # delta is the derivative of L along z
        delta = self.cost_derivative(output, y) * self.activation_func_derivative(zs[-1])
        dLdB[-1] = delta
        dLdW[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_func_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            dLdB[-l] = delta
            dLdW[-l] = np.dot(delta, activations[-l-1].transpose())

        return dLdB, dLdW

    def backprop(self, X, Y):
        """Calculate weight and bias derivatives for each neuron in each layer using (summing them up for each sample)
        samples X with answers Y via backpropagation.

        Args:
            X: Samples to train on, numpy ndarray. X has shape (self.N, M) where M is the number of samples
            Y: Answers to X. Y has shape (P, M) where P is the number of neurons in the last layer.

        Returns:
            A tuple (dLdB, dLdW) where dlDb contains derivatives of L (loss or cost function) along each bias and
            dLdW contains derivatives of L along each weight, summed up for each sample.
            Their shape is exactly the same as self.biases and self.weights -- list of numpy column vectors and
            list of numpy matrices.
        """
        assert (X.shape[0] == self.N)
        assert (Y.shape[0] == self.P)
        assert (X.shape[1] == Y.shape[1])
        dLdB = [np.zeros(b.shape) for b in self.biases]
        dLdW = [np.zeros(w.shape) for w in self.weights]

        # forward
        outputs, activations, zs = self.feedforward(X)

        # backward
        assert(zs[-1].shape == Y.shape)
        # delta is the derivative of L along z. delta's shape is (J, M), where J is number of neurons in a layer; J=P
        # at the moment
        delta = self.cost_derivative(outputs, Y) * self.activation_func_derivative(zs[-1])
        dLdB[-1] = delta
        dLdW[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_func_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            dLdB[-l] = delta
            dLdW[-l] = np.dot(delta, activations[-l-1].transpose())

        return dLdB, dLdW

    def grad_check_per_layer_per_minibatch(self, X, Y, layer_i, layer_dLdW_by_backprop):
        # max_measure = 0.0
        # for k in xrange(self.sizes[layer_i]):  # previous layer
        #     for j in xrange(self.sizes[layer_i + 1]):  # next layer
        #         dLdW_l_j_k_manually = self.calc_dLdW_manually(X, Y, layer_i, k, j)
        #         if layer_dLdW_by_backprop[j, k] == 0.0 and dLdW_l_j_k_manually == 0.0:  # to avoid division by zero
        #             continue
        #         measure = abs(layer_dLdW_by_backprop[j, k] - dLdW_l_j_k_manually) /\
        #                   abs(layer_dLdW_by_backprop[j, k] + dLdW_l_j_k_manually)
        #         if measure >= max_measure:
        #             max_measure = measure
        # print "grad check for layer {0}: max measure is {1}".format(layer_i, max_measure)
        if layer_i == 0:
            print "grad_check, layer {0} <backprop | manual>: {1} | {2}".format(layer_i,
                                                                            layer_dLdW_by_backprop[2, 1],
                                                                            self.calc_dLdW_manually(X, Y, layer_i, 1, 2))

    def calc_dLdW_manually(self, X, Y, l, k, j):
        """This function is similar to self.backprop in the sense that it calculates derivatives of L along weights.
        However it does it manually, using the definition of a derivative, and since vectorization here is not an
        option (we can't calculate more that one derivative per two passes), it does it in scalar way: it returns dL
        for a very specific w[l][j, k]. However, vectorization along multiple samples still work: we use here not just
        one sample x but a bunch of them and sum up derivatives for each of them to get the result.

        Args:
            X: Samples to train on, numpy ndarray. X has shape (self.N, M) where M is the number of samples.
            Y: Answers to X. Y has shape (P, M) where P is the number of neurons in the last layer.
            l: index of layer
            k: index of neuron in layer l
            j: index of nuron in layer l+1

        Returns:
            Scalar number dL/dw[l][j, k] calculated using the derivative definition.

        """
        saved_weight = self.weights[l][j, k]  # we will corrupt it while adding-subtracting eps

        eps = 0.0001
        self.weights[l][j, k] += eps
        C_right = self.cost(X, Y)
        self.weights[l][j, k] -= 1*eps
        C_left = self.cost(X, Y)

        self.weights[l][j, k] = saved_weight  # restore corrupted weights

        return (C_right - C_left) / (2*eps)

    def cost(self, X, Y):
        outputs = self.feedforward(X)[0]
        return self.cost_func(Y, outputs)


# cost functions
def msi(y, output_activations):
    """Calculate MSI.

    Args:
        y: correct answers. ndarray of shape (P, M) where M is the number of samples
           and P is the number of neurons in the last layer. This is exactly the same format as
           Network.feedforward()[0]
        output_activations: network's answers. ndarray of shape (P, M) where M is the number of samples
            and P is the number of neurons in the last layer. This is exactly the same format as
           Network.feedforward()[0]

    Returns:
        MSI cost, scalar value
    """
    assert (y.shape == output_activations.shape)
    # calculate msi for each sample, msi_per_sample shape is (M,)
    msi_per_sample = np.square(np.linalg.norm(y - output_activations, axis=0))
    # now sum them up and divide on /2M
    res = np.sum(msi_per_sample) / (1 * y.shape[1])
    print "msi per sample is {0}".format(msi_per_sample)
    print "res is {0}".format(res)
    return res

    # diff = y - output_activations
    # return np.dot(np.transpose(diff), diff)[0][0]


# returns a column vector -- derivative of C along all final activations
def msi_derivative(output_activations, y):
    return output_activations - y


# activation functions
def identity_activation(z):
    return z


def identity_activation_derivative(z):
    return np.ones(z.shape)


def relu(z):
    return np.maximum(z, 0)


def relu_derivative(z):
    return np.maximum(np.sign(z), 0)


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))


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
