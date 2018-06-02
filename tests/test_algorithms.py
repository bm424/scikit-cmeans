import pytest


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



