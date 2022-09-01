import os
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


def get_dataframe(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.dropna(inplace=True)
    data.drop(columns=["instance_weight"], inplace=True)
    data["income_over_50k"] = np.where(data["income_over_50k"] == " - 50000.", 0, 1)
    return data.apply(LabelEncoder().fit_transform)


def separate_features(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = [feature for feature in list(dataframe.columns) if feature != os.getenv("TARGET")]
    return dataframe[features], dataframe[os.getenv("TARGET")]


def main():
    load_dotenv()
    data_dir = Path(".").resolve() / "data"
    
    training_features, training_target = separate_features(get_dataframe(data_dir / os.getenv("TRAINING_DATA")))
    
    k_accuracies = []
    print("Training DecisionTreeClassifiers with variable max_depth")
    for k in range(2, 11):
        decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=k)
        decision_tree.fit(training_features, training_target)
        accuracy = decision_tree.score(training_features, training_target)
        k_accuracies.append((k, accuracy, decision_tree))
        print(f"{k:2} : {accuracy:.2%}")
    
    optimal_k = max(k_accuracies, key=lambda x: x[1])
    print(f"k depth with highest accuracy = {optimal_k[0]} ({optimal_k[1]:.2%})")

    k, training_accuracy, decision_tree = optimal_k[0], optimal_k[1], optimal_k[2]
    test_features, test_target = separate_features(get_dataframe(data_dir / os.getenv("TEST_DATA")))

    print(f"\nTesting DecisionTreeClassifier with optimal max_depth")

    test_accuracy = decision_tree.score(test_features, test_target)
    print(f"{k:2} : {test_accuracy:.2%}")

    all_features, all_target = pd.concat([training_features, test_features]), pd.concat([training_target, test_target])
    all_accuracy = decision_tree.score(all_features, all_target)
    is_overfitting = all_accuracy <= training_accuracy
    print("\nOverfitting Analysis")
    print(f"Training Acc: {training_accuracy:.2%}, Total Acc: {all_accuracy:.2%}")
    print(f"Overfitting?: {is_overfitting}{f' ({training_accuracy - all_accuracy:.2%})' if is_overfitting else ''}")
