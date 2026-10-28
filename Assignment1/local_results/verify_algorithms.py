"""Independent reference and edge-case checks for Assignment 1."""
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
for part in ("classification", "decision_tree", "k_nearest_neighbors"):
    sys.path.insert(0, str(ROOT / part))

import accuracy_error
import evaluation_metrics
from criterion import get_criterion_function
from knn_student import pairwise_dist, knn_predict, select_k_by_validation


def test_metrics_against_reference():
    rng = np.random.default_rng(42)
    truth, pred = rng.integers(0, 2, size=(2, 101))
    for name in ("accuracy_score", "precision_score", "recall_score", "f1_score"):
        module = accuracy_error if name == "accuracy_score" else evaluation_metrics
        assert np.isclose(getattr(module, name)(truth, pred),
                          getattr(metrics, name)(truth, pred), atol=1e-12)
    truth, pred = rng.normal(size=(2, 101))
    assert np.isclose(accuracy_error.mean_squared_error(truth, pred),
                      metrics.mean_squared_error(truth, pred))


def test_metrics_undefined_and_invalid_cases():
    for score in (evaluation_metrics.precision_score,
                  evaluation_metrics.recall_score, evaluation_metrics.f1_score):
        assert score([0, 0], [0, 0]) == 0.0
    for score in (accuracy_error.accuracy_score, accuracy_error.mean_squared_error):
        with pytest.raises(ValueError):
            score([], [])
        with pytest.raises(ValueError):
            score([1], [1, 2])


@pytest.mark.parametrize("name,expected", [
    ("info_gain", 2 / 3), ("info_gain_ratio", 2 / 3),
    ("gini", 2 / 9), ("error_rate", 1 / 3),
])
def test_multiclass_split_and_empty_child(name, expected):
    # Analytic three-class example with non-numeric labels.
    y = np.array(["a", "a", "b", "b", "c", "c"])
    score = get_criterion_function(name)
    assert np.isclose(score(y, y[:3], y[3:]), expected)
    assert score(y, y[:0], y) == 0.0
    assert score(y[:0], y[:0], y[:0]) == 0.0


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_distances_and_predictions_against_reference(metric):
    rng = np.random.default_rng(23300240009)
    train, query = rng.normal(size=(37, 5)), rng.normal(size=(13, 5))
    labels = rng.choice([-4, 2, 9], size=len(train))
    reference_metric = "euclidean" if metric == "l2" else "cosine"
    expected = metrics.pairwise_distances(query, train, metric=reference_metric)
    for mode in ("two_loops", "no_loops"):
        assert np.allclose(pairwise_dist(query, train, metric, mode), expected,
                           atol=1e-12)
        for k in (1, 3, 8, len(train)):
            reference = KNeighborsClassifier(n_neighbors=k, metric=reference_metric,
                                             algorithm="brute").fit(train, labels)
            assert np.array_equal(knn_predict(query, train, labels, k, metric, mode),
                                  reference.predict(query))


def test_cosine_zero_vectors_and_negative_label_tie():
    train = np.array([[0., 0.], [1., 0.], [-1., 0.]])
    expected = np.array([[1., 1., 1.], [1., 0., 2.]])
    assert np.allclose(pairwise_dist([[0., 0.], [1., 0.]], train,
                                     "cosine", "no_loops"), expected)
    assert knn_predict([[0., 0.]], [[1., 0.], [-1., 0.]], [9, -4],
                       2, "l2", "no_loops")[0] == -4


def test_unsorted_k_tie_and_invalid_k():
    train = np.array([[0.], [1.], [2.]])
    best, accs = select_k_by_validation(train, [0, 0, 0], [[0.5]], [0],
                                       [3, 1, 2], "l2", "no_loops")
    assert best == 1 and accs == [1.0, 1.0, 1.0]
    for k in (0, 4, 1.5, True):
        with pytest.raises(ValueError):
            knn_predict([[0.5]], train, [0, 0, 0], k, "l2", "no_loops")
