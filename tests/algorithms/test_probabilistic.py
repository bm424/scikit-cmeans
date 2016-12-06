"""Tests for the Probabilistic Fuzzy C-means algorithm.

"""

from unittest import TestCase
from unittest.mock import MagicMock
from nose_parameterized import parameterized

import numpy as np
from sklearn.datasets import make_blobs

from skcmeans.algorithms import Probabilistic
from skcmeans.initialization import initialize_random


class TestProbabilistic(TestCase):

    @parameterized.expand([
        ("2 clusters 2 features", 2, 2, (2, 2)),
        ("2 clusters 3 features", 2, 3, (2, 3)),
        ("3 clusters 2 features", 3, 2, (3, 2)),
        ("5 clusters 7 features", 5, 7, (5, 7))
    ])
    def test_first_update_assigns_center_shape(self, _, clusters, features, expected_shape):
        data, labels = make_blobs(centers=clusters, n_features=features)
        clusterer = Probabilistic(clusters)
        clusterer.update(data)
        self.assertEqual(clusterer.centers.shape, expected_shape)

    @parameterized.expand([
        ("2 clusters 2 features", 2, 2, (100, 2)),
        ("2 clusters 3 features", 2, 3, (100, 2)),
        ("3 clusters 2 features", 3, 2, (100, 3)),
        ("5 clusters 7 features", 5, 7, (100, 5))
    ])
    def test_first_update_assigns_membership_shape(self, _, clusters, features, expected_shape):
        data, labels = make_blobs(centers=clusters, n_features=features)
        clusterer = Probabilistic(clusters)
        clusterer.update(data)
        self.assertEqual(clusterer.memberships.shape, expected_shape)

    @parameterized.expand([
        ("2 clusters 2 features", 2, 2, (100, 2)),
        ("2 clusters 3 features", 2, 3, (100, 2)),
        ("3 clusters 2 features", 3, 2, (100, 3)),
        ("5 clusters 7 features", 5, 7, (100, 5))
    ])
    def test_distance_calculation_shape(self, _, clusters, features, expected):
        data, labels = make_blobs(centers=clusters, n_features=features)
        clusterer = Probabilistic(clusters)
        clusterer.update(data)
        self.assertEqual(clusterer.distances(data).shape, expected)

    @parameterized.expand([
        ("2 clusters 2 features", 2, 2, (100, 2)),
        ("2 clusters 3 features", 2, 3, (100, 2)),
        ("3 clusters 2 features", 3, 2, (100, 3)),
        ("5 clusters 7 features", 5, 7, (100, 5))
    ])
    def test_membership_calculation_shape(self, _, clusters, features, expected):
        data, labels = make_blobs(centers=clusters, n_features=features)
        clusterer = Probabilistic(clusters)
        clusterer.update(data)
        self.assertEqual(clusterer.calculate_memberships(data).shape, expected)

    @parameterized.expand([
        ("2 clusters 2 features", 2, 2),
        ("2 clusters 3 features", 2, 3),
        ("3 clusters 2 features", 3, 2),
        ("5 clusters 7 features", 5, 7)
    ])
    def test_membership_calculation_range(self, _, clusters, features):
        data, labels = make_blobs(centers=clusters, n_features=features)
        clusterer = Probabilistic(clusters)
        clusterer.converge(data)
        self.assertTrue(np.all(clusterer.memberships > 0) and np.all(clusterer.memberships < 1))

    @parameterized.expand([
        ("2 clusters 2 features", 2, 2, (2, 2)),
        ("2 clusters 3 features", 2, 3, (2, 3)),
        ("3 clusters 2 features", 3, 2, (3, 2)),
        ("5 clusters 7 features", 5, 7, (5, 7))
    ])
    def test_center_calculation_shape(self, _, clusters, features, expected):
        data, labels = make_blobs(centers=clusters, n_features=features)
        clusterer = Probabilistic(clusters)
        clusterer.update(data)
        self.assertEqual(clusterer.calculate_centers(data).shape, expected)

    @parameterized.expand([
        ("2 starts", 2),
        ("5 starts", 5),
        ("20 starts", 20)
    ])
    def test_fit_initialization_count(self, _, n_init):
        data, labels = make_blobs()
        clusterer = Probabilistic(2, n_init=n_init)
        clusterer.initialization = MagicMock(return_value=initialize_random(data, 2))
        clusterer.fit(data)
        self.assertEqual(clusterer.initialization.call_count, n_init)

    @parameterized.expand([
        ("2 starts", 2),
        ("5 starts", 5),
        ("20 starts", 20)
    ])
    def test_fit_converge_count(self, _, n_init):
        data, labels = make_blobs()
        clusterer = Probabilistic(2, n_init=n_init)
        clusterer.converge = MagicMock()
        clusterer.fit(data)
        self.assertEqual(clusterer.converge.call_count, n_init)

    @parameterized.expand([
        ("2 starts", 2),
        ("5 starts", 5),
        ("20 starts", 20)
    ])
    def test_fit_correct_assignment(self, _, n_init):
        data, labels = make_blobs()
        clusterer = Probabilistic(2, n_init=n_init)
        j_list = clusterer.fit(data)
        self.assertEqual(min(j_list), clusterer.objective(data))

    @parameterized.expand([
        ("2 iterations", 2),
        ("10 iterations", 10),
        ("100 iterations", 100)
    ])
    def test_converge_update_count(self, _, n_iter):
        data, labels = make_blobs()
        clusterer = Probabilistic(2, max_iter=n_iter)
        clusterer.update = MagicMock()
        clusterer.converge(data)
        self.assertLessEqual(clusterer.update.call_count, n_iter)


