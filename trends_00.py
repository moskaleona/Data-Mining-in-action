#!/usr/bin/env python
# -*- coding: utf8 -*-

####################################################
### You are not allowed to import anything else. ###
####################################################

import numpy as np


def power_sum(l, r, p=1.0):
    """
    input: l, r - integers, p - float
    returns sum of p powers of integers from [l, r]
    """
    powers = (np.arange(l, r + 1))**p
    return np.sum(powers)

def solve_equation(a, b, c):
    """
    input: a, b, c - integers
    returns float solutions x of the following equation: a x ** 2 + b x + c == 0
        In case of two diffrent solution returns tuple / list (x1, x2)
        In case of one solution returns one float
        In case of no float solutions return None
        In case of infinity number of solutions returns 'inf'
    """
    if a == b == c == 0:
        return 'inf'
    if a == b == 0:
        return
    if a == 0:
        return -c/b
    D = b ** 2 - 4 * a * c
    if D > 0:
        x1 = (-b + np.sqrt(D)) / (2 * a)
        x2 = (-b - np.sqrt(D)) / (2 * a)
        return (x1, x2)
    elif D == 0:
        return -b / (2 * a)
    else:
        return

def replace_outliers(x, std_mul=3.0):
    """
        input: x - numpy vector, std_mul - positive float
        returns copy of x with all outliers (elements, which are beyond std_mul * std from mean) replaced with mean
    """
    mean = np.mean(x)
    std = np.std(x)
    x_norm = x.copy()
    np.place(x_norm, abs(x - mean) > std_mul*std, mean)
    return x_norm


def get_eigenvector(A, alpha):
    """
    input: A - square numpy matrix, alpha - float
    returns numpy vector - eigenvector of A corresponding to eigenvalue alpha.
    """
    eps = 1e-8
    vals, vectors = np.linalg.eig(A)
    V = vectors[abs(vals-alpha)<=eps]
    if V.shape[0] == 0:
        return
    return V[0]


def discrete_sampler(p):
    """
        input: p - numpy vector of probability (non-negative, sums to 1)
        returns integer from 0 to len(p) - 1, each integer i is returned with probability p[i]
    """
    return np.random.choice(len(p),p=p)


def gaussian_log_likelihood(x, mu=0.0, sigma=1.0):
    """
        input: x - numpy vector, mu - float, sigma - positive float
        returns log p(x| mu, sigma) - log-likelihood of x dataset
        in univariate gaussian model with mean mu and standart deviation sigma
        """
    gaussian = np.exp(-(x - mu)**2 / (2 * (sigma**2))) / (np.sqrt(2 * np.pi) * sigma)
    return np.log(gaussian).sum()

def gradient_approx(f, x0, eps=1e-8):
    """
        input: f - callable, function of x. x0 - numpy vector, eps - float, represents step for x_i
        returns numpy vector - gradient of f in x0 calculated with finite difference method
        (for reference use https://en.wikipedia.org/wiki/Numerical_differentiation, search for "first-order divided difference")
    """
    E = np.eye(len(x0))
    gradf = np.zeros_like(x0, dtype = np.float)
    for i in range(len(x0)):
        gradf[i] = (f(x0 + eps*E[i].T) - f(x0))/eps
    return gradf

def gradient_method(f, x0, n_steps=1000, learning_rate=1e-2, eps=1e-8):
    """
        input: f - function of x. x0 - numpy vector, n_steps - integer, learning rate, eps - float.
        returns tuple (f^*, x^*), where x^* is local minimum point, found after n_steps of gradient descent,
                                        f^* - resulting function value.
        Impletent gradient descent method, given in the lecture.
        For gradient use finite difference approximation with eps step.
    """
    x_old = x0
    for _ in range(n_steps):
        x_new = x_old - learning_rate*gradient_approx(f,x_old,eps)
        x_old = x_new
    x_min = x_new
    f_min = f(x_min)
    return (f_min, x_min)

def linear_regression_predict(w, b, X):
    """
        input: w - numpy vector of M weights, b - bias, X - numpy matrix N x M (object-feature matrix),
        N - number of objects, M - number of features.
        returns numpy vector of predictions of linear regression model for X
        https://xkcd.com/1725/
    """
    y_pred = X.dot(w) + b
    return y_pred

def mean_squared_error(y_true, y_pred):
    """
        input: two numpy vectors of object targets and model predictions.
        return mse
    """
    return 1./len(y_pred)*np.sum((y_pred-y_true)**2)

def linear_regression_mse_gradient(w, b, X, y_true):
    """
        input: w, b - weights and bias of a linear regression model,
                X - object-feature matrix, y_true - targets.
        returns gradient of linear regression model mean squared error w.r.t (with respect to) w and b
    """
    y_pred = linear_regression_predict(w, b, X)
    grad_mse = np.zeros(len(w) + 1, dtype=np.float)
    for i in range(len(w)):
        grad_mse[i] = 2./len(w)*np.sum((y_pred-y_true)*X[:,i])
    grad_mse[-1] = 2./len(w)*np.sum(y_pred-y_true)
    return grad_mse


class LinearRegressor:
    def fit(self, X_train, y_train, n_steps=1000, learning_rate=1e-2, eps=1e-8):
        """
            input: object-feature matrix and targets.
            optimises mse w.r.t model parameters
        """
        for _ in range(n_steps):
            grad = linear_regression_mse_gradient(self.w, self.b, X_train, y_train)
            self.w -= learning_rate*grad[:-1]
            self.b -= learning_rate*grad[-1]
        return self


    def predict(self, X):
        return linear_regression_predict(self.w, self.b, X)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_der(x):
    """
    returns sigmoid derivative w.r.t. x
    """
    return sigmoid(x)*(1-sigmoid(x))


def relu(x):
    return np.maximum(x, 0)


def relu_der(x):
    """
        return relu (sub-)derivative w.r.t x
    """
    if x >= 0:
        return 1
    else:
        return 0



class MLPRegressor:
    """
    simple dense neural network class for regression with mse loss.
    """
    def __init__(self, n_units=[32, 32], nonlinearity=relu):
        """
            input: n_units - number of neurons for each hidden layer in neural network,
                    nonlinearity - activation function applied between hidden layers.
        """
        self.n_units = n_units
        self.nonlinearity = nonlinearity


    def fit(self, X_train, y_train, n_steps=1000, learning_rate=1e-2, eps=1e-8):
        """
            input: object-feature matrix and targets.
            optimises mse w.r.t model parameters
            (you may use approximate gradient estimation)
        """
        raise NotImplementedError


    def predict(self, X):
        """
            input: object-feature matrix
            returns MLP predictions in X
        """
        raise NotImplementedError



