from loader import MNISTLoader
from network import Network
import numpy as np
import argparse


def evaluate_mnist():
    loader = MNISTLoader()
    net = Network([784, 30, 10])

    net.SGD(loader.train_data, loader.train_target_vectorized, epochs=1, mini_batch_size=10, eta=1.0)
    net.evaluate_mnist(loader.test_data, loader.test_target)


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
    [0.0]
])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="<xor|mnist>")
    parser.add_argument("-s", "--sizes", nargs='+', help="Layer sizes", type=int)

    parser.add_argument("-gch", "--gradcheck", help="Log derivatives dL/dW[l][j, k]. Don't forget to specify also"
                                                    " -gchl, -gchj, -gchk", action="store_true")
    parser.add_argument("-gchl", type=int)
    parser.add_argument("-gchj", type=int)
    parser.add_argument("-gchk", type=int)

    parser.add_argument("-act", "--activation", help="Activation function, <identity|relu|sigmoid>, by default"
                                                     " the latter")

    parser.add_argument("-ep", "--epochs", help="Number of epochs", type=int, default=1)
    parser.add_argument("-mbs", "--mini_batch_size", help="Mini batch size", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=1.0)
    args = parser.parse_args()
    network = Network(args.sizes, activation=args.activation, gch=args.gradcheck, gchl=args.gchl, gchj=args.gchj,
                      gchk=args.gchk)
    if args.task == "xor":
        network.SGD(xor_train_data, xor_train_target,
                    epochs=args.epochs, mini_batch_size=args.mini_batch_size, eta=args.learning_rate)
        network.evaluate_xor(xor_train_data, xor_train_target)
    # evaluate_mnist()