from skcmeans.algorithms import CMeans

import numpy as np


def test_initialize(algorithm: CMeans):
    assert algorithm._initialized is False
    algorithm.cluster_centers_ = np.array([(0, 1), (0, -1)])
    assert algorithm._initialized is True


def test_first_update(algorithm, blobs):
    data, labels = blobs
    n_samples, n_features = data.shape
    n_clusters = algorithm.n_clusters

    print('n_samples:', n_samples)
    print('n_features:', n_features)
    print('n_clusters:', n_clusters)

    algorithm.update(data)
    assert algorithm.cluster_centers_.shape, (n_clusters, n_features)
    assert algorithm.memberships_.shape, (n_samples, n_clusters)
    assert algorithm.distances(data).shape == (n_samples, n_clusters)
    assert algorithm.calculate_memberships(data).shape == (n_samples, n_clusters)
    assert algorithm.calculate_centers(data).shape == (n_clusters, n_features)


def test_converge(algorithm, blobs):
    data, labels = blobs
    results = algorithm.converge(data)
    assert results['n_iter'] >= algorithm.max_iter or results['objective_delta'] < algorithm.tol


def test_predict(algorithm: CMeans, blobs):
    data, labels = blobs
    algorithm.fit(data)
    prediction = algorithm.predict(data)
    assert prediction.shape == labels.shape
    assert set(np.unique(prediction)) <= set(range(algorithm.n_clusters))



