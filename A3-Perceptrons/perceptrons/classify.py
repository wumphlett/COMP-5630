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

from sklearn.linear_model import Perceptron


def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    perceptron = Perceptron(eta0=learning_rate, max_iter=max_iter)
    perceptron.fit(train_set, train_labels)
    return list(perceptron.coef_[0]), perceptron.intercept_[0]


def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    perceptron = Perceptron(eta0=learning_rate, max_iter=max_iter)
    perceptron.fit(train_set, train_labels)
    return list(perceptron.predict(dev_set))
