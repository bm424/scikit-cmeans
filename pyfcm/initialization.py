from sklearn.utils import check_random_state

from pyfcm import algorithms


def initialize_random(x, k, random_state=None):
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
    selection : :class:`np.ndarray`
        (k, n_features)
        A length-k subset of the original data.

    """
    n_samples = x.shape[0]
    seeds = check_random_state(random_state).permutation(n_samples)[:k]
    selection = x[seeds]
    return selection


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
        (k, n_features)
        A length-k subset of the original data.

    """

    clusterer = algorithms.Probabilistic(n_clusters=k, random_state=random_state)
    clusterer.converge(x)
    return clusterer.centers
