# Feed-Forward Neural Network (FFNN) for Regression Problems

## Reference

- Mathematical background: ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/index.html).

- Datasets: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php).

## Characteristics

- The code has been written and tested in Python 3.7.7.
- Multi-input/multi-output (multivariate) feed-forward neural network implementation for regression problems.
- Arbitrary number of nodes for input, hidden, and output layers.
- Continuous problem: quadratic cost function, L2-type regularization term, sigmoid activation for hidden layers, linear activation for output layer.
- Classification problem: cross-entropy cost function, L2-type regularization term, sigmoid activation for hidden and output layers, classes determined automatically.
- Gradient of the cost function calculated using the backpropagation algorithm.
- Option to reduce the learning rate during the computation.
- Option to not to compute and return the gradient.
- A gradient descent optimizer (GDO) is included in *utils.py*, together with several other utility functions.
- The *FFNN* class in *FFNN.py* is not constrained to the GDO solver but it can be used with any other optimizer.
- Usage: *python test.py example*.

## Parameters

`example` Name of the example to run (wine, stock, wifi, pulsar.)

`problem` Defines the type of problem. Equal to C specifies a classification problem, anything else specifies a continuous problem. The default value is `None`.

`use_grad` Specifies if the gradient is calculated and returned. The default value is `True`.

`init_weights` Specifies if the weights are randomly initialized. The default value is `True`.

`data_file` File name with the dataset (csv format).

`n_features` Number of features in the dataset (needed only for continuous problems).

`hidden_layers` List, tuple, or array with the number of nodes in each hidden layers.

`split_factor` Split value between training and test data.

`L2` Regularization factor.

`epochs` Max. number of iterations (GDO).

`alpha` Learning rate (GDO).

`d_alpha` Rate decay of the learning rate (GDO).

`tolX`, `tolF` Gradient absolute tolerance and function relative tolerance (GDO). If both are specified the GDO will exit if either is satisfied. If both are not specified the GDO will exit when the max. number of iterations is reached.

## Examples

There are four examples in *test.py*: wine, stock, wifi, pulsar. Since GDO is used, `use_grad` is set to `True`. For all examples `init_weights` is also set to `True`.

### Single-label continuous problem example: wine

```python
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
```

Original dataset: <https://archive.ics.uci.edu/ml/datasets/Wine+Quality>.

The dataset has 11 features, 1 label, and 4898 samples.

The neural network has a layout of [11, 20, 1] and 261 variables.

Correlation predicted/actual values: 0.708 (training), 0.601 (test).

Exit on `epochs` with `tolX` = 2.0e-4 and `tolF` = 1.1e-7.

### Multi-label continuous problem example: stock

```python
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
```

Original dataset: <https://archive.ics.uci.edu/ml/datasets/ISTANBUL+STOCK+EXCHANGE>.

The dataset has 6 features, 3 labels, and 536 samples.

The neural network has a layout of [6, 4, 4, 3] and 63 variables.

Correlation predicted/actual values: 0.841 (training), 0.840 (test).

Exit on `epochs` with `tolX` = 4.7e-6 and `tolF` = 9.8e-11.

### Multi-class classification problem example: wifi

```python
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
```

Original dataset: <https://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization>.

The dataset has 7 features, 4 classes, and 2000 samples.

The neural network has a layout of [7, 10, 5, 4, 4] and 179 variables.

Accuracies predicted/actual values: 100.0% (training), 98.0% (test).

Exit on `epochs` with `tolX` = 3.9e-5 and `tolF` = 1.0e-8.

### Multi-class classification problem example: pulsar

```python
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
```

Original dataset: <https://archive.ics.uci.edu/ml/datasets/HTRU2>.

The dataset has 8 features, 2 classes, and 17898 samples.

The neural network has a layout of [8, 10, 10, 2] and 222 variables.

Accuracies predicted/actual values: 98.1% (training), 98.0% (test).

Exit on `epochs` with `tolX` = 2.5e-4 and `tolF` = 5.5e-7.
