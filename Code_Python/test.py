"""
Feed-Forward Neural Net (FFNN) for Regression Problems

Copyright (c) 2020 Gabriele Gilardi


References
----------
- Mathematical background: "Neural Networks and Deep Learning" @
http://neuralnetworksanddeeplearning.com/index.html.

- Datasets: UCI Machine Learning Repository @
https://archive.ics.uci.edu/ml/datasets.php.

Characteristics
---------------
- The code has been written and tested in Python 3.7.7.
- Multi-input/multi-output (multivariate) feed-forward neural network
  implementation for regression problems.
- Arbitrary number of nodes for input, hidden, and output layers.
- Continuous problem: quadratic cost function, L2-type regularization term,
  sigmoid activation for hidden layers, linear activation for output layer.
- Classification problem: cross-entropy cost function, L2-type regularization
  term, sigmoid activation for hidden and output layers, classes determined
  automatically.
- Gradient of the cost function calculated using the backpropagation algorithm.
- Option to reduce the learning rate during the computation.
- Option to not to compute and return the gradient.
- A gradient descent optimizer (GDO) is included in <utils.py., together
  with several other utility functions.
- The <FFNN> class in <FFNN.py> is not constrained to the GDO solver but it
  can be used with any other optimizer.
- Usage: python test.py <example>.

Parameters
----------
example = house, stock, seed, wine
    Name of the example to run.
problem
    Defines the type of problem. Equal to C specifies specifies a
    classification problem, anything else specifies a continuous problem.
    The default value is <None>.
use_grad = True, False
    Specifies if the gradient is calculated and returned. The default value
    is <True>.
init_weights = True, False
    Specifies if the weights are randomly initialized. The default value is
    <True>.
data_file
    File name with the dataset (csv format).
n_features
    Number of features in the dataset (needed only for continuous problems).
hidden_layers
    List, tuple, or array with the number of nodes in each hidden layers.
0 < split_factor < 1
    Split value between training and test data.
L2
    Regularization factor.
epochs
    Max. number of iterations (GDO).
0 < alpha <= 1
    Learning rate (GDO).
0 < d_alpha <= 1
    Rate decay of the learning rate (GDO).
tolX, tolF
    Gradient absolute tolerance and function relative tolerance (GDO). If both
    are specified the GDO will exit if either is satisfied. If both are not
    specified the GDO will exit when the max. number of iterations is reached.
"""

import sys
import numpy as np
import FFNN as nnet
import utils as utl

# Read example to run
if len(sys.argv) != 2:
    print("Usage: python test.py <example>")
    sys.exit(1)
example = sys.argv[1]

problem = None                  # By default is a continuous problem
use_grad = True                 # By default calculate and return the gradient
init_weights = True             # By default randomly initialize weights
np.random.seed(1294404794)

#  Single-label continuous problem example
if (example == 'wine'):
    # Dataset: 11 features, 1 label, 4898 samples
    # Neural network: layout of [11, 20, 1], 261 variables
    # Correlation predicted/actual values: 0.708 (training), 0.601 (test)
    # Exit on epochs with tolX = 2.0e-4 and tolF = 1.1e-7
    # https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    data_file = 'wine_dataset.csv'
    n_features = 11
    hidden_layers = [20]
    split_factor = 0.7
    L2 = 0.0
    epochs = 50000
    alpha = 0.2
    d_alpha = 1.0
    tolX = 1.e-7
    tolF = 1.e-7

#  Multi-label continuous problem example
elif (example == 'stock'):
    # Dataset: 6 features, 3 label, 536 samples
    # Neural network: layout of [6, 4, 4, 3], 63 variables
    # Correlation predicted/actual values: 0.841 (training), 0.840 (test)
    # Exit on epochs with tolX = 4.7e-6 and tolF = 9.8e-11
    # https://archive.ics.uci.edu/ml/datasets/ISTANBUL+STOCK+EXCHANGE
    data_file = 'stock_dataset.csv'
    n_features = 6
    hidden_layers = [4, 4]
    split_factor = 0.70
    L2 = 0.0
    epochs = 50000
    alpha = 0.99
    d_alpha = 1.0
    tolX = 1.e-7
    tolF = 1.e-15

# Multi-class classification problem example
elif (example == 'wifi'):
    # Dataset: 7 features, 4 classes, 2000 samples
    # Neural network: layout of [7, 10, 5, 4, 4], 179 variables
    # Accuracies predicted/actual values: 100.0% (training), 98.0% (test).
    # Exit on epochs with tolX = 3.9e-5 and tolF = 1.0e-8
    # https://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization
    data_file = 'wifi_dataset.csv'
    problem = 'C'
    hidden_layers = [10, 5, 4]
    split_factor = 0.70
    L2 = 0.0
    epochs = 50000
    alpha = 0.9
    d_alpha = 1.0
    tolX = 1.e-7
    tolF = 1.e-10

# Multi-class classification problem example
elif (example == 'pulsar'):
    # Dataset: 8 features, 2 classes, 17898 samples
    # Neural network: layout of [8, 10, 10, 2], 222 variables
    # Accuracies predicted/actual values: 98.1% (training), 98.0% (test).
    # Exit on epochs with tolX = 2.5e-4 and tolF = 5.5e-7
    # https://archive.ics.uci.edu/ml/datasets/HTRU2
    data_file = 'pulsar_dataset.csv'
    problem = 'C'
    hidden_layers = [10, 10]
    split_factor = 0.70
    L2 = 0.0
    epochs = 5000
    alpha = 0.9
    d_alpha = 1.0
    tolX = 1.e-7
    tolF = 1.e-7

else:
    print("Example not found")
    sys.exit(1)

# Read data from a csv file
data = np.loadtxt(data_file, delimiter=',')
n_samples, n_cols = data.shape

# Logistic regression (the label column is always the last one)
if (problem == 'C'):
    n_features = n_cols - 1
    n_labels = 1
    n_outputs, class_list = utl.get_classes(data[:, -1])

# Linear regression (the label columns are always at the end)
else:
    n_labels = n_cols - n_features
    n_outputs = n_labels

n_inputs = n_features
layout = np.block([n_inputs, np.asarray(hidden_layers), n_outputs])

# Total number of variables
n_var = 0
for i in range(1, len(layout)):
    n_var += layout[i] * (layout[i-1] + 1)

# Randomly build the training (tr) and test (te) datasets
rows_tr = int(split_factor * n_samples)
rows_te = n_samples - rows_tr
idx_tr = np.random.choice(np.arange(n_samples), size=rows_tr, replace=False)
idx_te = np.delete(np.arange(n_samples), idx_tr)
data_tr = data[idx_tr, :]
data_te = data[idx_te, :]

# Split the data
X_tr = data_tr[:, 0:n_features]
Y_tr = data_tr[:, n_features:]
X_te = data_te[:, 0:n_features]
Y_te = data_te[:, n_features:]

# Info
print("\nNumber of samples = ", n_samples)
print("Number of features = ", n_features)
print("Number of labels = ", n_labels)

print("\nNumber of inputs (NN) = ", n_inputs)
print("Number of outputs (NN) = ", n_outputs)
print("Number of variables = ", n_var)

print("\nNN layout = ", layout)
if (problem == 'C'):
    print("\nClasses: ", class_list)

print("\nNumber of training samples = ", rows_tr)
print("Number of test samples= ", rows_te)

# Normalize training dataset
Xn_tr, param = utl.normalize_data(X_tr)

# Initialize learner
learner = nnet.FFNN(layout, L2=L2, problem=problem, use_grad=use_grad,
                    init_weights=init_weights)

# Gradient descent optimization
func = learner.create_model
theta0 = np.zeros(n_var)        # If init_weight = True the values are ignored
args = (Xn_tr, Y_tr)
theta, F, info = utl.GDO(func, theta0, args=args, epochs=epochs, alpha=alpha,
                         d_alpha=d_alpha, tolX=tolX, tolF=tolF)

# Results
print("\nCoeff. = ")
print(theta)
print("F = ", F)
print("Info = ", info)

# Evaluate training data
Yp_tr = learner.eval_data(Xn_tr)

# Normalize and evaluate test data
Xn_te = utl.normalize_data(X_te, param)
Yp_te = learner.eval_data(Xn_te)

# Results for logistic regression (accuracy and correlation)
if (problem == 'C'):
    print("\nAccuracy training data = ", utl.calc_accu(Yp_tr, Y_tr))
    print("Corr. training data = ", utl.calc_corr(Yp_tr, Y_tr))
    print("\nAccuracy test data = ", utl.calc_accu(Yp_te, Y_te))
    print("Corr. test data = ", utl.calc_corr(Yp_te, Y_te))

# Results for linear regression (RMSE and correlation)
else:
    print("\nRMSE training data = ", utl.calc_rmse(Yp_tr, Y_tr))
    print("Corr. training data = ", utl.calc_corr(Yp_tr, Y_tr))
    print("\nRMSE test data = ", utl.calc_rmse(Yp_te, Y_te))
    print("Corr. test data = ", utl.calc_corr(Yp_te, Y_te))
