import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math

# Function that : Creates a list of random minibatches from (X, Y)
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):

    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    inc = mini_batch_size

    # Partition (shuffled_X, shuffled_Y).
    # Cases with a complete mini batch size only i.e each of 64 examples.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * inc: (k + 1) * inc]
        mini_batch_Y = shuffled_Y[:, k * inc: (k + 1) * inc].reshape((1, inc))

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # For handling the end case (last mini-batch <


if m % mini_batch_size != 0:
    mini_batch_X = shuffled_X[:, int(num_complete_minibatches * inc): int(
        (num_complete_minibatches * inc) + (m - (num_complete_minibatches * (m / inc))))]
    mini_batch_Y = shuffled_Y[:, int(num_complete_minibatches * inc): int(
        (num_complete_minibatches * inc) + (m - (num_complete_minibatches * (m / inc))))]

    mini_batch = (mini_batch_X, mini_batch_Y)
    mini_batches.append(mini_batch)

return mini_batches

# Testing teh Algorithm
np.random.seed(1)
mini_batch_size = 64
nx = 12288
m = 148
X = np.array([x for x in range(nx * m)]).reshape((m, nx)).T
Y = np.random.randn(1, m) < 0.5

mini_batches = random_mini_batches(X, Y, mini_batch_size)