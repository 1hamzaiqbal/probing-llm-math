from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np

def make_Xy(features_by_problem, labels_by_problem, layer_idx):
    # features_by_problem: dict[id] -> np.array[L, D]
    # labels_by_problem: dict[id] -> topic_label (int)
    ids = sorted(set(features_by_problem) & set(labels_by_problem))
    X = np.stack([features_by_problem[i][layer_idx] for i in ids])
    y = np.array([labels_by_problem[i] for i in ids])
    return X, y, ids

def train_probe(X_tr, y_tr):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    return clf
