import argparse
import time
import numpy as np

from . import reader
from . import classify as c

"""
This file contains the main application that is run for this MP.
"""


def compute_accuracies(predicted_labels, dev_labels):
    if not isinstance(predicted_labels, list):
        print("Predict labels must be list")
        assert False
    yhats = predicted_labels
    if len(yhats) != len(dev_labels):
        print("Predict labels must have the same length as actual labels!")
        assert False
    if not all([(y == 0 or y == 1) for y in yhats]):
        print("Predicted labels must only contain 0s or 1s")
        assert False
    accuracy = np.mean(yhats == dev_labels)
    tp = np.sum([yhats[i] == dev_labels[i] and yhats[i] == 1 for i in range(len(yhats))])
    precision = tp / np.sum([yhats[i] == 1 for i in range(len(yhats))])
    recall = tp / (np.sum([yhats[i] != dev_labels[i] and yhats[i] == 0 for i in range(len(yhats))]) + tp)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("Accuracy:", accuracy)
    print("F1-Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)

    return accuracy, f1, precision, recall


def main():
    parser = argparse.ArgumentParser(description="MP5 Classify")

    parser.add_argument(
        "--dataset", dest="dataset_file", type=str, default="data", help="the directory of the training data"
    )
    parser.add_argument("--method", default="perceptron", help="classification method, ['perceptron']")
    parser.add_argument("--lrate", dest="lrate", type=float, default=1e-2, help="Learning rate - default 1.0")
    parser.add_argument("--max_iter", dest="max_iter", type=int, default=10, help="Maximum iterations - default 10")

    args = parser.parse_args()

    start_time = time.time()

    if args.method == "perceptron":
        train_set, train_labels, dev_set, dev_labels = reader.load_dataset(args.dataset_file)
        pred_p = c.classifyPerceptron(train_set, train_labels, dev_set, args.lrate, args.max_iter)
        print("Perceptron")
        accuracy, f1, precision, recall = compute_accuracies(pred_p, dev_labels)
    else:
        print("Method must be perceptron!")

    end_time = time.time()

    runtimes = {"perceptron": 1.18, "knn": 1.67}

    print(f"Time taken for {args.method} to execute training and classification: {end_time - start_time}")
    print(f"Time taken for our model implementation is approximately {runtimes[args.method]}")


if __name__ == "__main__":
    main()
