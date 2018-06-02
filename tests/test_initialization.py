import pytest
from unittest import TestCase, mock

import numpy as np

from skcmeans.initialization import initialize_random, initialize_probabilistic


@pytest.mark.parametrize('k', [2, 3, 5, 6, 7, 9, 15])
@pytest.mark.parametrize('initializer', [
    initialize_random,
    initialize_probabilistic
])
def test_initialize_random(blobs, k, initializer):
    data, labels = blobs
    n_samples, n_features = data.shape
    distance, centers = initializer(data, k)
    assert distance.shape == (n_samples, k)
    assert centers.shape == (k, n_features)



