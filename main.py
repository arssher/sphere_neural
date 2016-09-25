from loader import MNISTLoader
from network import Network
import numpy as np


if __name__ == "__main__":
    # loader = MNISTLoader(train_percent=0.1, test_percent=0.1)
    # net = Network([784, 10, 10])
    # net.SGD(loader.train_data, loader.train_target_vectorized, 1, 10, 3.0)
    # net.evaluate_mnist(loader.train_target, loader.train_target)

    # xor_train_data = np.array([
    #     [0, 0, 1, 1],
    #     [0, 1, 0, 1]
    # ])
    # xor_train_target = np.array([
    #     [0.0, 1.0, 1.0, 0.0],
    # ])

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