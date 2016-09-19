from loader import MNISTLoader
from network import Network


if __name__ == "__main__":
    loader = MNISTLoader(train_percent=0.01, test_percent=0.01)
    # loader = MNISTLoader(train_percent=0.5, test_percent=0.1)
    net = Network([784, 30, 10])
    net.SGD(loader.train_data, loader.train_target_vectorized, 1, 5, 3.0)
    net.evaluate(loader.train_data, loader.test_target)