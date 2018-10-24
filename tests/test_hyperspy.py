import pytest
from sklearn.datasets import make_blobs
import hyperspy.api as hs

from skcmeans.hyperspy import *
from skcmeans.algorithms import *


@pytest.fixture(params=[2, 3])
def blob_data(request):
    return make_blobs(n_samples=16, n_features=16, shuffle=False,
                      centers=request.param)


@pytest.fixture(params=[
    (hs.signals.Signal1D, (16, 16)),
    (hs.signals.Signal1D, (4, 4, 16)),
    (hs.signals.Signal2D, (16, 4, 4)),
    (hs.signals.Signal2D, (4, 4, 4, 4)),
])
def signal(request, blob_data):
    signal_type, shape = request.param
    data, _ = blob_data
    return signal_type(data.reshape(shape))


@pytest.mark.parametrize('n_clusters', [
    2, 3, 4
])
def test_cluster_number(signal, n_clusters):
    cluster(signal, n_clusters)
    assert signal.learning_results.memberships.shape == \
           (signal.axes_manager.navigation_size, n_clusters, )
    assert signal.learning_results.centers.shape == \
           (n_clusters, signal.axes_manager.signal_size, )


@pytest.mark.parametrize('algorithm_param, Algorithm', [
    ('hard', Hard,),
    ('probabilistic', Probabilistic,),
    ('possibilistic', Possibilistic,),
    ('probabilisticgk', ProbabilisticGK),
])
def test_cluster_algorithm(signal, algorithm_param, Algorithm):
    alg = cluster(signal, algorithm=algorithm_param)
    assert isinstance(alg, Algorithm)


@pytest.mark.parametrize('output_dimension', [2, 3, 4, 5])
def test_use_decomposition_results(signal, output_dimension):
    # Decompose and use it
    signal.decomposition(output_dimension=output_dimension)
    alg = cluster(signal, use_decomposition_results=True, reproject=True)
    expected_shape = (alg.n_clusters, signal.axes_manager.signal_size,)
    assert signal.learning_results.centers.shape == expected_shape

    # Decompose, don't reproject
    alg = cluster(signal, use_decomposition_results=True, reproject=False)
    expected_shape = (alg.n_clusters, output_dimension,)
    assert signal.learning_results.centers.shape == expected_shape

    # Decompose, don't use it
    alg = cluster(signal, use_decomposition_results=False, reproject=True)
    expected_shape = (alg.n_clusters, signal.axes_manager.signal_size,)
    assert signal.learning_results.centers.shape == expected_shape

    # Don't decompose, try to use it
    signal.learning_results.loadings = None
    with pytest.raises(ValueError):
        alg = cluster(signal, use_decomposition_results=True, reproject=False)

    # Don't decompose, don't use it
    alg = cluster(signal, use_decomposition_results=False, reproject=True)
    expected_shape = (alg.n_clusters, signal.axes_manager.signal_size,)
    assert signal.learning_results.centers.shape == expected_shape


def test_get_cluster_centers(signal):
    alg = cluster(signal)
    centers = get_cluster_centers(signal)
    assert centers.axes_manager.navigation_shape == (alg.n_clusters,)
    assert centers.axes_manager.signal_shape == \
           signal.axes_manager.signal_shape


def test_get_cluster_memberships(signal):
    alg = cluster(signal)
    memberships = get_cluster_memberships(signal)
    assert memberships.axes_manager.navigation_shape == (alg.n_clusters,)
    assert memberships.axes_manager.signal_shape == \
           signal.axes_manager.navigation_shape

