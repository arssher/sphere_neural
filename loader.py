from sklearn.datasets import fetch_mldata
from utils import shuffle_in_unison
import numpy as np

# TODO: fix target shape. Right now it is samples_number x 10
class MNISTLoader(object):
    def __init__(self, train_percent=0.8, test_percent=None):
        mnist = fetch_mldata('MNIST original')
        data, target = shuffle_in_unison(mnist.data, mnist.target)
        total_samples_number = mnist.data.shape[0]
        self.train_samples_number = int(total_samples_number * train_percent)
        self.test_samples_number = total_samples_number - self.train_samples_number
        if test_percent is not None:
            self.test_samples_number = min(self.test_samples_number, int(total_samples_number * test_percent))
        print "Train data contains %s samples, test data contains %s samples" %\
              (self.train_samples_number, self.test_samples_number)

        self.train_data = data[:self.train_samples_number]
        self.train_target = target[:self.train_samples_number]

        self.train_target_vectorized = np.empty((self.train_samples_number, 10))
        for i in xrange(self.train_samples_number):
            self.train_target_vectorized[i] = vectorize_mnist_target(self.train_target[i])

        self.test_data = data[self.train_samples_number:self.train_samples_number + self.test_samples_number]
        self.test_target = target[self.train_samples_number:self.train_samples_number + self.test_samples_number]


# turns 2 into np.array([0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0]), 9 into np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]), etc
def vectorize_mnist_target(i):
    e = np.zeros(10)
    e[i] = 1.0
    return e
