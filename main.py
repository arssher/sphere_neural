from loader import MNISTLoader
from network import Network


if __name__ == "__main__":
    loader = MNISTLoader()
    net = Network([784, 10])
    net.evaluate(loader.test_data, loader.test_target)