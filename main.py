from loader import MNISTLoader
from network import Network
import numpy as np
import argparse

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
    parser.add_argument("-act", "--activation", help="Activation function, <identity|relu|sigmoid>, by default"
                                                     " sigmoid")
    parser.add_argument("-c", "--cost", help="Cost function, <mse|ce>, by default mse")

    parser.add_argument("-gch", "--gradcheck", help="Log manually computed derivatives to compare them with calculated"
                                                    "via backprop", action="store_true")
    parser.add_argument("-gch_eps", "--gradcheck_eps", help="increment to compute derivatives manually", type=float)

    parser.add_argument("-ep", "--epochs", help="Number of epochs, by default 1", type=int, default=1)
    parser.add_argument("-mbs", "--mini_batch_size", help="Mini batch size, by default 1", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate, by default 1.0", type=float, default=1.0)
    args = parser.parse_args()
    network = Network(args.sizes, activation=args.activation, cost=args.cost, gradient_check=args.gradcheck,
                      gradient_check_eps=args.gradcheck_eps)
    if args.task == "xor":
        network.SGD(xor_train_data, xor_train_target,
                    epochs=args.epochs, mini_batch_size=args.mini_batch_size, eta=args.learning_rate)
        network.evaluate_xor(xor_train_data, xor_train_target)
    elif args.task == "mnist":
        loader = MNISTLoader()
        network.SGD(loader.train_data, loader.train_target_vectorized,
                    epochs=args.epochs, mini_batch_size=args.mini_batch_size, eta=args.learning_rate)
        network.evaluate_mnist(loader.test_data, loader.test_target)
    # evaluate_mnist()

# TODO list:
# grad check description
# add cross entropy
# softmax
# initial wieghts
# add regularization
# compare comparison? how? findings?
# mod grad / iteration number
# different learning rate?
# debug relu?
# move task evaluation outside?

# Questions:
# summing up derivatives per each sample?
# just derive MSE analytically?
