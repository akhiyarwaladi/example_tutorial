import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))


def KL_divergence(x, y):
    return x * np.log(x/y) + (1-x) * np.log((1-x)/(1-y))


# visible_size: the number of input units (probably 64)
# hidden_size: the number of hidden units (probably 25)
# lambda_: weight decay parameter
# sparsity_param: The desired average activation for the hidden units (denoted in the lecture
#                            notes by the greek alphabet rho, which looks like a lower-case "p").
# beta: weight of sparsity penalty term
# data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.
#
# The input theta is a vector (because minFunc expects the parameters to be a vector).
# We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
# follows the notation convention of the lecture notes.
# Returns: (cost,gradient) tuple
def sparse_autoencoder_cost(theta, visible_size, hidden_size,
                            lambda_, sparsity_param, beta, data):

    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    W1 = theta[0:hidden_size*visible_size].reshape(hidden_size, visible_size)
    W2 = theta[hidden_size*visible_size:2*hidden_size*visible_size].reshape(visible_size, hidden_size)
    b1 = theta[2*hidden_size*visible_size:2*hidden_size*visible_size+hidden_size]
    b2 = theta[2*hidden_size*visible_size+hidden_size:]

    # Number of training examples
    m = data.shape[1]

    # Forward propagation
    z2 = W1.dot(data) + np.tile(b1, (m,1)).transpose()
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + np.tile(b2, (m,1)).transpose()
    h = sigmoid(z3)

    # Sparsity
    rho_hat = np.sum(a2, axis=1)
    rho = np.tile(sparsity_param, hidden_size)


    # Cost function
    cost = np.sum((h - data) ** 2) / (2 * m) + \
           (lambda_ / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2)) + \
           (beta / m) * np.sum(KL_divergence(rho_hat, rho))



    # Backprop
    sparsity_delta = np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (m,1)).transpose()

    delta3 = -(data - h) * sigmoid_prime(z3)
    delta2 = (W2.transpose().dot(delta3) + beta * sparsity_delta) * sigmoid_prime(z2)
    W1grad = delta2.dot(data.transpose())
    W2grad = delta3.dot(a2.transpose())
    b1grad = np.sum(delta2, axis=1)
    b2grad = np.sum(delta3, axis=1)

    # After computing the cost and gradient, we will convert the gradients back
    # to a vector format (suitable for minFunc).  Specifically, we will unroll
    # your gradient matrices into a vector.
    grad = np.concatenate((W1grad.reshape(hidden_size* visible_size),
                           W2grad.reshape(hidden_size*visible_size),
                           b1grad.reshape(hidden_size),
                           b2grad.reshape(visible_size)))

    return (cost, grad)