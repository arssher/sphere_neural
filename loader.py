from sklearn.datasets import fetch_mldata
from utils import shuffle_in_unison
import numpy as np
import cPickle
import gzip
from sklearn.preprocessing import normalize


class MNISTLoader(object):
    def __init__(self, train_percent=0.8, test_percent=None):
        mnist = fetch_mldata('MNIST original')
        data = normalize(mnist.data.astype(np.float64))
        target = np.around(mnist.target).astype(np.int8)
        data, target = shuffle_in_unison(data, target)
        total_samples_number = data.shape[0]
        self.train_samples_number = int(total_samples_number * train_percent)
        self.test_samples_number = total_samples_number - self.train_samples_number
        if test_percent is not None:
            self.test_samples_number = min(self.test_samples_number, int(total_samples_number * test_percent))
        print "Train data contains %s samples, test data contains %s samples" %\
              (self.train_samples_number, self.test_samples_number)

        self.train_data = data[:self.train_samples_number]
        self.train_target = target[:self.train_samples_number]

        self.train_target_vectorized = np.empty((self.train_data.shape[0], 10))
        for i in xrange(self.train_data.shape[0]):
            self.train_target_vectorized[i] = vectorize_mnist_target(self.train_target[i])

        self.test_data = data[self.train_samples_number:self.train_samples_number + self.test_samples_number]
        self.test_target = target[self.train_samples_number:self.train_samples_number + self.test_samples_number]

# turns 2 into np.array([0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0]), 9 into np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]), etc
def vectorize_mnist_target(i):
    e = np.zeros(10)
    e[i] = 1.0
    return e
