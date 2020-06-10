"""
Utility functions for regression problems.

Copyright (c) 2020 Gabriele Gilardi
"""

import numpy as np


def GDO(func, X0, epochs=500, args=None, alpha=0.9, d_alpha=1.0, tolX=None,
        tolF=None):
    """
    Gradient Descent Optimization

    func            Function to minimize
    epochs          Number of iterations
    alpha           Learning rate
    d_alpha         Rate decay of the learning rate
    tolX            Tolerance on the gradient
    tolF            Relative tolerance on the function
    args            Tuple containing any parameter that needs to be passed to
                    the function
    """
    # Initialize
    X = X0
    F_old = np.finfo(float).max
    deltaX = None
    deltaF = None

    # Main loop
    for epoch in range(epochs):

        # Evaluate the function and calculate next value for X
        F, grad = func(X, args)
        X = X - alpha * grad

        # Check termination criteria on the gradient
        if (tolX is not None):
            deltaX = np.amax(np.abs(grad))
            if (deltaX < tolX):
                break

        # Check termination criteria on the function
        if (tolF is not None):
            deltaF = np.abs(F_old - F) / (1.0 + np.abs(F_old))
            if (deltaF < tolF):
                break

        # Next epoch
        F_old = F
        alpha = alpha * d_alpha

    # Evaluate the function before to exit and return convergency info
    F, grad = func(X, args)
    info = ('epoch', epoch, 'tolX', deltaX, 'tolF', deltaF)
    return X, F, info


def normalize_data(X, param=(), ddof=0):
    """
    If mu and sigma are not defined, returns a column-normalized version of
    X with zero mean and standard deviation equal to one. If mu and sigma are
    defined returns a column-normalized version of X using mu and sigma.

    X           Input dataset
    Xn          Column-normalized input dataset
    param       Tuple with mu and sigma
    mu          Mean
    sigma       Standard deviation
    ddof        Delta degrees of freedom (if ddof = 0 then divide by m, if
                ddof = 1 then divide by m-1, with m the number of data in X)
    """
    # Column-normalize using mu and sigma
    if (len(param) > 0):
        Xn = (X - param[0]) / param[1]
        return Xn

    # Column-normalize using mu=0 and sigma=1
    else:
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=ddof)
        Xn = (X - mu) / sigma
        param = (mu, sigma)
        return Xn, param


def scale_data(X, param=()):
    """
    If X_min and X_max are not defined, returns a column-scaled version of
    X in the interval (-1,+1). If X_min and X_max are defined returns a
    column-scaled version of X using X_min and X_max.

    X           Input dataset
    Xs          Column-scaled input dataset
    param       Tuple with X_min and X_max
    X_min       Min. value along the columns of X
    X_max       Max. value along the columns of X
    """
    # Column-scale using X_min and X_max
    if (len(param) > 0):
        Xs = -1.0 + 2.0 * (X - param[0]) / (param[1] - param[0])
        return Xs

    # Column-scale using X_min=-1 and X_max=+1
    else:
        X_min = np.amin(X, axis=0)
        X_max = np.amax(X, axis=0)
        Xs = -1.0 + 2.0 * (X - X_min) / (X_max - X_min)
        param = (X_min, X_max)
        return Xs, param


def get_classes(Y):
    """
    Returns the number of classes (unique values) in array Y and the
    corresponding list.
    """
    class_list = np.unique(Y)
    n_classes = len(class_list)

    return n_classes, class_list


def build_class_matrix(Y):
    """
    Builds the output array Yout for a classification problem. Array Y has
    dimensions (n_samples, 1) while Yout has dimension (n_samples, n_classes).
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


def regression_sol(X, Y):
    """
    Returns the closed-form solution to the linear regression problem.

    X           (n_samples, 1+n_features)           Input dataset
    Y           (n_samples, n_labels)               Output dataset
    theta       (1+n_features, n_labels)            Regression parameters

    Notes:
    - the input dataset must include the column of 1s.
    - each COLUMN contains the coefficients of each output label.
    """
    theta = np.linalg.pinv(X.T @ X) @ X.T @ Y

    return theta


def calc_rmse(a, b):
    """
    Returns the root-mean-square-error of arrays <a> and <b>. If the arrays
    are multi-column, the RMSE is calculated as all the columns are one
    single vector.
    """
    a = a.flatten()
    b = b.flatten()
    rmse = np.sqrt(((a - b) ** 2).sum() / len(a))

    return rmse


def calc_corr(a, b):
    """
    Returns the correlation between arrays <a> and <b>. If the arrays are
    multi-column, the correlation is calculated as all the columns are one
    single vector.
    """
    a = a.flatten()
    b = b.flatten()
    corr = np.corrcoef(a, b)[0, 1]

    return corr


def calc_accu(a, b):
    """
    Returns the accuracy (in %) between arrays <a> and <b>.The two arrays must
    be column/row vectors.
    """
    a = a.flatten()
    b = b.flatten()
    accu = 100.0 * (a == b).sum() / len(a)

    return accu
