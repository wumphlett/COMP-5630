"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np


class Perceptron:
    def __init__(self, learning_rate: float = 1e-2, max_iter: int = 10):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None

    def fit(self, train_set: np.ndarray, train_labels: np.ndarray):
        m, n = train_set.shape
        self.weights = np.zeros((n + 1, 1))

        for iteration in range(self.max_iter):
            for i, x in enumerate(train_set):

                x = np.append(x, 1).reshape(-1, 1)
                y_predicted = 1. if np.squeeze(np.dot(x.T, self.weights)) >= 0 else 0.

                if y_predicted != train_labels[i]:
                    self.weights += self.learning_rate * ((1. if train_labels[i] else -1.) * x)

        return self.weights[:-1], self.weights[-1]

    def predict(self, test_set: np.ndarray):
        labels = []

        for i, x in enumerate(test_set):
            x = np.append(x, 1).reshape(-1, 1)
            labels.append(True if np.squeeze(np.dot(x.T, self.weights)) >= 0 else False)

        return labels


def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    perceptron = Perceptron(learning_rate=learning_rate, max_iter=max_iter)
    perceptron.fit(train_set, train_labels)
    weights = perceptron.weights.flatten()
    return list(weights[:-1]), weights[-1]


def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    perceptron = Perceptron(learning_rate=learning_rate, max_iter=max_iter)
    perceptron.fit(train_set, train_labels)
    return perceptron.predict(dev_set)
