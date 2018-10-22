import pytest
from sklearn.datasets import make_blobs
import hyperspy.api as hs
import numpy as np

from skcmeans.hyperspy import cluster
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
    2, 3
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
])
def test_cluster_algorithm(signal, algorithm_param, Algorithm):
    alg = cluster(signal, algorithm=algorithm_param)
    assert isinstance(alg, Algorithm)

