"""
Module that contains our new MLP architecture.
"""

import torch
from torch import nn

import numpy as np

from sklearn.neural_network import MLPClassifier


def drop_in(
    arr: "np.ndarray",
    drop_in_rate=0.1,
    drop_in_with_value=1,
    seed=None,
):
    """
    Performs drop in
    >>> arr = np.zeros((10,10))
    >>> drop_in(arr,seed=123)
    """
    if seed is not None:
        np.random.seed(seed)
    return arr + np.random.binomial(drop_in_with_value, drop_in_rate, size=arr.shape)


class MLPSkl:
    def __init__(
        self,
        perform_drop_in=False,
    ):
        self.clf = MLPClassifier(
            hidden_layer_sizes=(
                128,
                128,
            ),
            max_iter=2000,
        )
        self.perform_drop_in = perform_drop_in
        self.use_tanh = False

    def phase(self):
        return "train"

    def _featurize(self, X):
        if isinstance(X, list):
            X = np.vstack([np.array(x) for x in X])
        if self.phase() == "train" and self.perform_drop_in:
            X = drop_in(X)

        if self.use_tanh:
            X = np.tanh(X)
        return X

    def fit(self, X, y):
        X = self._featurize(X)
        return self.clf.fit(X, y)

    def predict(self, X):
        X = self._featurize(X)
        return self.clf.predict(X)

    def predict_proba(self, X):
        X = self._featurize(X)
        return self.clf.predict_proba(X)
