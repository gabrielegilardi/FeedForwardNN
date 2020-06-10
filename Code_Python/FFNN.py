"""
Feed-Forward Neural Network (FFNN) Class

Copyright (c) 2020 Gabriele Gilardi


X           (n_samples, n_inputs)           Input dataset (training)
Xp          (n_samples, n_inputs)           Input dataset (prediction)
Y           (n_samples, n_outputs)          Output dataset (training)
Yp          (n_samples, n_labels)           Output dataset (prediction)
L2          scalar                          Regularization factor
J           scalar                          Cost function
grad        (n_var, )                       Unrolled gradient
theta       (n_var, )                       Unrolled weights

n_samples           Number of samples
n_inputs            Number of nodes in the input layer (also number of
                    features in the original input dataset)
n_outputs           Number of nodes in the output layer (also number of
                    labels/classes in the output dataset)
n_labels            Number of outputs in the original dataset
n_var               Number of variables
hidden_layers       Number of nodes in the hidden layers
n_layers            Number of layers

Layout structure:
equal to            [n_inputs, hidden_layers, n_outputs]
size of             (n_layers, )
layout[i]           Number of nodes of the (i)th layer
layout[i-1]         Number of inputs of the (i)th layer
layout[i-1]+1       Number of inputs of the (i)th layer plus bias term

Layer components:
Z           (n_samples, nodes)          Weighted inputs
A           (n_samples, nodes)          Activations
D           (n_samples, nodes)          Delta errors
W           (nodes, 1+inputs)           Weights (includes the bias terms)
grad        (nodes, 1+inputs)           Gradient of the cost function

nodes       Layer number of nodes.
inputs      Layer number of inputs (also number of nodes previous layer)

Notes:
- the weights associated with the bias are in W[:, 0], while the weights
  associated with the inputs are in W[:, 1:].
- the number of variables in each layer is <nodes * (inputs + 1)>.
"""

import numpy as np


def f_activation(z):
    """
    Numerically stable version of the sigmoid function (reference:
    http://fa.bianp.net/blog/2019/evaluate_logistic/#sec3.)
    """
    a = np.zeros_like(z)

    idx = (z >= 0.0)
    a[idx] = 1.0 / (1.0 + np.exp(-z[idx]))

    idx = np.invert(idx)
    a[idx] = np.exp(z[idx]) / (1.0 + np.exp(z[idx]))

    return a


def f1_activation(z):
    """
    Derivative of activation function (sigmoid).
    """
    a = f_activation(z) * (1.0 - f_activation(z))
    return a


def logsig(z):
    """
    Numerically stable version of the log-sigmoid function (reference:
    http://fa.bianp.net/blog/2019/evaluate_logistic/#sec3.)
    """
    a = np.zeros_like(z)

    idx = (z < -33.3)
    a[idx] = z[idx]

    idx = (z >= -33.3) & (z < -18.0)
    a[idx] = z[idx] - np.exp(z[idx])

    idx = (z >= -18.0) & (z < 37.0)
    a[idx] = - np.log1p(np.exp(-z[idx]))

    idx = (z >= 37.0)
    a[idx] = - np.exp(-z[idx])

    return a


def build_class_matrix(Y):
    """
    Builds the output array <Yout> for a classification problem. Array <Y> has
    dimensions (n_samples, 1) and <Yout> has dimension (n_samples, n_classes).
    Yout[i,j] = 1 specifies that the i-th sample belongs to the j-th class.
    """
    n_samples = Y.shape[0]

    # Classes and corresponding number
    Yu, idx = np.unique(Y, return_inverse=True)
    n_classes = len(Yu)

    # Build the array actually used for classification
    Yout = np.zeros((n_samples, n_classes))
    Yout[np.arange(n_samples), idx] = 1.0

    return Yout, Yu


class layer:

    def __init__(self, nodes, inputs=0):
        """
        nodes                 # Number of nodes
        inputs                # Number of inputs (nodes previous layer)
        """
        self.nodes = nodes
        self.inputs = inputs

        self.Z = np.array([])               # Weighted inputs
        self.A = np.array([])               # Activations
        self.D = np.array([])               # Delta errors
        self.W = np.array([])               # Weights (includes the bias terms)
        self.grad = np.array([])            # Gradient of the cost function


class FFNN:

    def __init__(self, layout, L2=0.0, problem=None, use_grad=True,
                 init_weights=True):
        """
        layout          Neural network layout
        L2              Regolarization factor
        problem         C = logistic regression, otherwise linear regression
        use_grad        True = calculate and return the gradient
        init_weight     True = randomly initialize the weights
        """
        self.layout = np.asarray(layout)
        self.problem = problem
        self.init_weights = init_weights
        self.use_grad = use_grad
        self.L2 = L2

        self.n_inputs = self.layout[0]              # Number of features
        self.n_outputs = self.layout[-1]            # Number of classes/labels
        self.n_layers = len(self.layout)            # Number of layers

        # Initialize layers
        self.NN = [layer(self.layout[0])]
        for i in range(1, self.n_layers):
            self.NN.append(layer(self.layout[i], layout[i-1]))

        # For logistic regression only
        if (self.problem == 'C'):
            self.init_Y = True                      # Flag to initialize Yout
            self.Yout = np.array([])                # Actual output
            self.Yu = np.array([])                  # Class list

    def create_model(self, theta, args):
        """
        Creates the model for a linear/logistic regression problem.
        """
        # Unpack
        X = args[0]                 # Input dataset
        Y = args[1]                 # Output dataset

        # Build the weights
        self.build_weights(theta)

        # If requested randomly initialize the weights
        if (self.init_weights):
            self.set_init_weights()
            self.init_weights = False

        # Feed-forward step
        self.feed_forward(X)

        # Logistic regression problem
        if (self.problem == 'C'):

            # The first time initialize Yout (output) and Yu (class list)
            if (self.init_Y):
                self.Yout, self.Yu = build_class_matrix(Y)
                self.init_Y = False

            # Cross-entropy cost function and gradient
            J, grad = self.cross_entropy_function()

        # Linear regression problem
        else:

            # Quadratic cost function and gradient
            self.NN[-1].A = self.NN[-1].Z           # Output layer is linear
            J, grad = self.quadratic_function(Y)

        # If not used don't return the gradient
        if (self.use_grad):
            return J, grad
        else:
            return J

    def eval_data(self, Xp):
        """
        Evaluates the input dataset with the model created in <create_model>.
        """
        # Feed-forward step
        self.feed_forward(Xp)

        # Logistic regression problem
        if (self.problem == 'C'):

            # Most likely class
            idx = np.argmax(self.NN[-1].A, axis=1)
            Yp = self.Yu[idx].reshape((len(idx), 1))

        # Linear regression problem
        else:

            # Output layer is linear
            Yp = self.NN[-1].Z

        return Yp

    def build_weights(self, theta):
        """
        Builds the weights from the array of variables. Each layer weight
        matrix has size (nodes, 1 +inputs).
        """
        idx1 = 0
        for i in range(1, self.n_layers):
            n_rows = self.NN[i].nodes
            n_cols = 1 + self.NN[i].inputs
            idx2 = idx1 + n_rows * n_cols
            self.NN[i].W = np.reshape(theta[idx1:idx2], (n_rows, n_cols))
            idx1 = idx2

    def set_init_weights(self):
        """
        If requested randomly sets the initial weights.
        """
        for i in range(1, self.n_layers):
            n_rows = self.NN[i].nodes
            n_cols = self.NN[i].inputs

            # Weights associated with the bias term
            self.NN[i].W[:, 0] = np.random.normal(0., 1., n_rows)

            # Weights associated with the inputs from the previous layer
            self.NN[i].W[:, 1:] = np.random.normal(0., 1., (n_rows, n_cols)) \
                                  / np.sqrt(float(n_cols))

    def feed_forward(self, X):
        """
        Carries out the feed forward step.
        """
        B = np.ones((X.shape[0], 1))        # Bias contribution
        self.NN[0].A = X.copy()             # Input layer
        for i in range(1, self.n_layers):
            self.NN[i].Z = np.hstack((B, self.NN[i-1].A)) @ (self.NN[i].W).T
            self.NN[i].A = f_activation(self.NN[i].Z)

    def cross_entropy_function(self):
        """
        Computes the cross-entropy cost function and the gradient (including
        the L2 regularization term).
        """
        n_samples = self.Yout.shape[0]

        # Cost function  (activation value is calculated in the logsig function)
        error = (1.0 - self.Yout) * self.NN[-1].Z - logsig(self.NN[-1].Z)
        J = error.sum() / n_samples
        for i in range(1, self.n_layers):
            J += self.L2 * (self.NN[i].W[:, 1:] ** 2.0).sum() / (2.0 * n_samples)

        # Return if gradient is not requested
        if (self.use_grad is False):
            return J, 0.0

        # Delta errors
        self.NN[-1].D = self.NN[-1].A - self.Yout
        for i in range(self.n_layers-1, 1, -1):
            self.NN[i-1].D = (self.NN[i].D @ self.NN[i].W[:, 1:]) \
                              * f1_activation(self.NN[i-1].Z)

        # Gradient
        B = np.ones((n_samples, 1))            # Bias contribution
        for i in range(1, self.n_layers):
            V = np.zeros((self.NN[i].nodes, 1))
            self.NN[i].grad = \
                ((self.NN[i].D).T @ np.hstack((B, self.NN[i-1].A))
                + self.L2 * np.hstack((V, self.NN[i].W[:, 1:]))) / n_samples

        # Unroll the gradient
        grad = np.array([])
        for i in range(1, self.n_layers):
            grad = np.concatenate((grad, self.NN[i].grad), axis=None)

        # Return the cost function and the unrolled gradient
        return J, grad

    def quadratic_function(self, Y):
        """
        Computes the quadratic cost function and the gradient (including the
        L2 regularization term).
        """
        n_samples = Y.shape[0]

        # Cost function
        J = ((self.NN[-1].A - Y) ** 2.0).sum() / (2.0 * n_samples)
        for i in range(1, self.n_layers):
            J += self.L2 * (self.NN[i].W[:, 1:] ** 2.0).sum() / (2.0 * n_samples)

        # Return if gradient is not requested
        if (self.use_grad is False):
            return J, 0.0

        # Delta errors
        self.NN[-1].D = self.NN[-1].A - Y
        for i in range(self.n_layers-1, 1, -1):
            self.NN[i-1].D = (self.NN[i].D @ self.NN[i].W[:, 1:]) \
                              * f1_activation(self.NN[i-1].Z)

        # Gradient
        B = np.ones((n_samples, 1))            # Bias contribution
        for i in range(1, self.n_layers):
            V = np.zeros((self.NN[i].nodes, 1))
            self.NN[i].grad = \
                ((self.NN[i].D).T @ np.hstack((B, self.NN[i-1].A))
                + self.L2 * np.hstack((V, self.NN[i].W[:, 1:]))) / n_samples

        # Unroll the gradient
        grad = np.array([])
        for i in range(1, self.n_layers):
            grad = np.concatenate((grad, self.NN[i].grad), axis=None)

        # Return the cost function and the unrolled gradient
        return J, grad
