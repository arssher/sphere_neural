from loader import MNISTLoader
from network import Network
import numpy as np


def evaluate_mnist():
    loader = MNISTLoader()
    net = Network([784, 30, 10])

    net.SGD(loader.train_data, loader.train_target_vectorized, epochs=1, mini_batch_size=10, eta=3.0)
    net.evaluate_mnist(loader.test_data, loader.test_target)


def evaluate_xor():
    xor_train_data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    xor_train_target = np.array([
        [0.0],
        [1.0],
        [1.0],
        [1.0]
    ])

    net = Network([2, 4, 1])
    net.SGD(xor_train_data, xor_train_target, 1000, 1, 0.5)
    net.evaluate_xor(xor_train_data, xor_train_target)

if __name__ == "__main__":
    evaluate_mnist()
