import os
from pathlib import Path
import time
from typing import List, Optional

from dotenv import load_dotenv
import pandas as pd
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB


def get_dataframe(path: Path, names: Optional[List[str]] = None) -> pd.DataFrame:
    if names:
        return pd.read_csv(path, names=names)
    return pd.read_csv(path, header=None)


def train_model(naive_bayes_implementation, train, train_labels, test, test_labels, name):
    nb = naive_bayes_implementation()

    print(f"Training {name} Naive Bayes")
    start = time.time()
    nb.fit(train, train_labels.values.ravel())
    end = time.time()

    print(f"Training Acc: {nb.score(train, train_labels.values.ravel()):.2%}")
    print(f"Testing Acc: {nb.score(test, test_labels.values.ravel()):.2%}")
    print(f"Execution Time: {end - start:.3f}s\n")


def main():
    load_dotenv()
    data_dir = Path(".").resolve() / "data"

    with open(data_dir / os.getenv("NAMES")) as name_file:
        column_names = [name for name in name_file.read().split("\n")]

    training_features = get_dataframe(data_dir / os.getenv("TRAINING_DATA"), names=column_names)
    training_labels = get_dataframe(data_dir / os.getenv("TRAINING_LABELS"))

    test_features = get_dataframe(data_dir / os.getenv("TEST_DATA"), names=column_names)
    test_labels = get_dataframe(data_dir / os.getenv("TEST_LABELS"))

    print("Training and Testing Multiple Naive Bayes Implementations")
    train_model(MultinomialNB, training_features, training_labels, test_features, test_labels, "Multinomial")
    train_model(GaussianNB, training_features, training_labels, test_features, test_labels, "Gaussian")
    train_model(ComplementNB, training_features, training_labels, test_features, test_labels, "Complement")
    train_model(BernoulliNB, training_features, training_labels, test_features, test_labels, "Bernoulli")
