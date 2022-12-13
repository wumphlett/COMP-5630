"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        return 1. / (1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        m, n = X_train.shape
        self.weights = np.zeros((n + 1, 1))

        X_train = np.insert(X_train, n, 1, axis=1)

        for _ in range(self.epochs):
            labels = []
            for x in X_train:
                labels.append(self.boundary(x))
            predictions = np.array(labels)
            gradient = np.dot(X_train.T, predictions - y_train)
            gradient = np.divide(gradient, n + 1)
            gradient = np.multiply(gradient, self.lr)
            for i, adjustment in enumerate(gradient):
                self.weights[i] -= adjustment

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        '''
        labels = []

        for i, x in enumerate(X_test):
            x = np.append(x, 1).reshape(-1, 1)
            labels.append(True if np.squeeze(np.dot(x.T, self.weights)) >= 0 else False)

        return np.array(labels)
        '''
        labels = []
        for x in X_test:
            x = np.append(x, 1)
            labels.append(self.boundary(x))
        return np.array(labels)

    def boundary(self, features):
        return 1 if self.sigmoid(np.dot(features, self.weights)) >= self.threshold else 0

