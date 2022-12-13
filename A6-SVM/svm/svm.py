"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # Will be set in train/pred
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.
        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.
        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C
        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        N, D = X_train.shape
        dist = 1 - y_train * np.dot(X_train, self.w)

        updates = self.reg_const * y_train[:,np.newaxis] * X_train
        updates = (dist > 0).astype(int)[:,np.newaxis] * updates
        updates = -updates + self.w

        return np.sum(updates, axis=0)/N

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.
        Hint: operate on mini-batches of data for SGD.
        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        y_train = np.where(y_train == 0, -1, 1)
        X_train = np.append(np.ones((X_train.shape[0],1)), X_train, axis=1)

        N, D = X_train.shape
        BATCH_SIZE = N
        self.w = np.zeros(D)
        for _ in range(self.epochs):
            for i in range(N // BATCH_SIZE):
                X_curr, y_curr = X_train[i*BATCH_SIZE : (i+1)*BATCH_SIZE],  y_train[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                self.w -= self.alpha * self.calc_gradient(X_curr, y_curr)
        return

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
        X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)

        pred = np.dot(X_test, self.w)
        return (pred > 0).astype(int)
