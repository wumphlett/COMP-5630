"""Data preprocessing."""

import os
import pickle
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_pickle(f: str) -> Any:
    """Load a pickle file.

    Parameters:
        f: the pickle filename

    Returns:
        the pickled data
    """
    return pickle.load(f, encoding="latin1")


def load_CIFAR_batch(filename: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single batch of cifar data.

    Parameters:
        filename: the pickle filename

    Returns:
        the data
        the labels
    """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all of cifar data.

    Parameters:
        ROOT: the root directory containing the data

    Returns:
        training data
        training labels
        testing data
        testing labels
    """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_{}".format(b))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(
    num_training: int = 49000,
    num_validation: int = 1000,
    num_test: int = 10000,
    subtract_mean: bool = True,
):
    """Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.

    Parameters:
        num_training: number of training images
        num_validation: number of validation images
        num_test: number of test images
        subtract_mean: whether or not to normalize the data

    Returns:
        the train/val/test data and labels
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = os.path.join("cifar10", "cifar-10-batches-py")
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def get_MUSHROOM_data(validation: float, testing: float = 0.2) -> dict:
    """Load the mushroom dataset.

    Parameters:
        validation: portion of the dataset used for validation
        testing: portion of the dataset used for testing

    Returns
        the train/val/test data and labels
    """
    X_train = np.load("mushroom/X_train.npy")
    y_train = np.load("mushroom/y_train.npy")
    y_test = np.load("mushroom/y_test.npy")
    X_test = np.load("mushroom/X_test.npy")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation / (1 - testing), random_state=123
    )
    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    return data


def construct_MUSHROOM():
    """Convert raw categorical data from mushroom dataset to one-hot encodings."""
    dataset = pd.read_csv("mushroom/mushrooms.csv")
    y = dataset["class"]
    X = dataset.drop("class", axis=1)
    Encoder_X = LabelEncoder()
    for col in X.columns:
        X[col] = Encoder_X.fit_transform(X[col])
    Encoder_y = LabelEncoder()
    y = Encoder_y.fit_transform(y)
    X = X.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    np.save("mushroom/X_train.npy", X_train)
    np.save("mushroom/y_train.npy", y_train)
    np.save("mushroom/X_test.npy", X_test)
    np.save("mushroom/y_test.npy", y_test)
