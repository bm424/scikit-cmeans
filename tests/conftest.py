import pytest
import itertools
from sklearn.datasets import make_blobs
from skcmeans.algorithms import Hard, Probabilistic, Possibilistic

algorithm_param_data = {
    'n_clusters': [1, 3],
    'max_iter': [1, 10],
    'tol': [1, 1e-2],
    'metric': ['euclidean', 'cityblock', 'cosine']
}

blob_param_data = {
    'n_features': [2, 5],
    'n_samples': [50, 100],
    'centers': [1, 3, 8],
}


def combine_param_data(param_data):
    values_list = itertools.product(*param_data.values())
    return [dict(zip(param_data.keys(), v)) for v in values_list]

algorithm_params = combine_param_data(algorithm_param_data)
blob_params = combine_param_data(blob_param_data)


@pytest.fixture(params=blob_params)
def blobs(request):
    data, labels = make_blobs(**request.param)
    return data, labels


@pytest.fixture(params=[Hard, Probabilistic, Possibilistic])
def algorithm_class(request):
    return request.param


@pytest.fixture(params=algorithm_params)
def algorithm(request, algorithm_class):
    return algorithm_class(**request.param)