from scipy.spatial.distance import cdist
import numpy as np

from sklearn.utils import check_random_state

from . import algorithms



def initialize_random(x, k, random_state=None, eps=1e-12):
    """Selects initial points randomly from the data.

    Parameters
    ----------
    x : :class:`np.ndarray`
        (n_samples, n_features)
        The original data.
    k : int
        The number of points to select.
    random_state : int or :class:`np.random.RandomState`, optional
        The generator used for initialization. Using an integer fixes the seed.

    Returns
    -------
    Unitialized memberships_
    selection : :class:`np.ndarray`
        (k, n_features)
        A length-k subset of the original data.

    """
    n_samples = x.shape[0]
    seeds = check_random_state(random_state).permutation(n_samples)[:k]
    selection = x[seeds] + eps
    distances = cdist(x, selection)
    normalized_distance = distances / np.sum(distances, axis=1)[:, np.newaxis]
    return 1-normalized_distance, selection


def initialize_probabilistic(x, k, random_state=None):
    """Selects initial points using a probabilistic clustering approximation.

    Parameters
    ----------
    x : :class:`np.ndarray`
        (n_samples, n_features)
        The original data.
    k : int
        The number of points to select.
    random_state : int or :obj:`np.random.RandomState`, optional
        The generator used for initialization. Using an integer fixes the seed.

    Returns
    -------
    :class:`np.ndarray`
        (n_samples, k)
        Cluster memberships_
    :class:`np.ndarray`
        (k, n_features)
        Cluster cluster_centers_

    """

    clusterer = algorithms.Probabilistic(n_clusters=k, random_state=random_state)
    clusterer.converge(x)
    return clusterer.memberships_, clusterer.cluster_centers_
