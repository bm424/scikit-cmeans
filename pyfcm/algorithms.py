"""Implementations of a number of C-means algorithms.

References
----------
.. [1] J. C. Bezdek, J. Keller, R. Krisnapuram, and N. R. Pal, Fuzzy models
   and algorithms for pattern recognition and image processing. Kluwer Academic
   Publishers, 2005.

"""

import numpy as np
from scipy.spatial.distance import cdist
from .initialization import initialize_random, initialize_probabilistic


class CMeans:
    """Base class for C-means algorithms.

    Parameters
    ----------
    n_clusters : int, optional
        The number of clusters to find.
    n_init : int, optional
        The number of times to attempt convergence with new initial centroids.
    max_iter : int, optional
        The number of cycles of the alternating optimization routine to run for
        *each* convergence.
    tol : float, optional
        The stopping condition. Convergence is considered to have been reached
        when the objective function changes less than `tol`.
    verbosity : int, optional
        The verbosity of the instance. May be 0, 1, or 2.
        .. note:: Very much not yet implemented.
    random_state : :obj:`int` or :obj:`np.random.RandomState`, optional
        The generator used for initialization. Using an integer fixes the seed.

    Attributes
    ----------
    metric : :obj:`string` or :obj:`function`
        The distance metric used. May be any of the strings specified for
        :obj:`cdist`, or a user-specified function.
    initialization : function
        The method used to initialize the cluster centers.
    centers : :obj:`np.ndarray`
        (n_clusters, n_features)
        The derived or supplied cluster centers.
    memberships : :obj:`np.ndarray`
        (n_samples, n_clusters)
        The derived or supplied cluster memberships.

    """

    metric = 'euclidean'
    initialization = staticmethod(initialize_random)

    def __init__(self, n_clusters=2, n_init=10, max_iter=300, tol=1e-4,
                 verbosity=0, random_state=None, **kwargs):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbosity
        self.random_state = random_state
        self.params = kwargs
        self.centers = None
        self.memberships = None

    def distances(self, x):
        """Calculates the distance between data x and the centers.

        The distance, by default, is calculated according to `metric`, but this
        method should be overridden by subclasses if required.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        Returns
        -------
        :obj:`np.ndarray`
            (n_samples, n_clusters)
            Each entry (i, j) is the distance between sample i and cluster
            center j.

        """
        return cdist(x, self.centers, metric=self.metric)

    def calculate_memberships(self, x):
        raise NotImplementedError(
            "`calculate_memberships` should be implemented by subclasses.")

    def calculate_centers(self, x):
        raise NotImplementedError(
            "`calculate_centers` should be implemented by subclasses.")

    def objective(self, x):
        raise NotImplementedError(
            "`objective` should be implemented by subclasses.")

    def fit(self, x):
        """Optimizes cluster centers by restarting convergence several times.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        """
        objective_best = np.infty
        memberships_best = None
        centers_best = None
        for i in range(self.n_init):
            self.converge(x)
            objective = self.objective(x)
            if objective < objective_best:
                memberships_best = self.memberships.copy()
                centers_best = self.centers.copy()
                objective_best = objective
        self.memberships = memberships_best
        self.centers = centers_best

    def converge(self, x):
        """Finds cluster centers through an alternating optimization routine.

        Terminates when either the number of cycles reaches `max_iter` or the
        objective function changes by less than `tol`.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        """
        j_new = self.objective(x)
        for i in range(self.max_iter):
            j_old = j_new
            self.update(x)
            j_new = self.objective(x)
            if j_old - j_new < self.tol:
                break

    def update(self, x):
        """Updates cluster memberships and centers in a single cycle.

        If the cluster centers have not already been initialized, they are
        chosen according to `initialization`.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        """
        if self.centers is not None:
            self.centers = self.calculate_centers(x)
        else:
            self.centers = self.initialization(x, self.n_clusters,
                                               self.random_state)
        self.memberships = self.calculate_memberships(x)


class Hard(CMeans):
    """Hard C-means, equivalent to K-means clustering.

    Methods
    -------
    calculate_memberships(x)
        The membership of a sample is 1 to the closest cluster and 0 otherwise.
    calculate_centers(x)
        New centers are calculated as the mean of the points closest to them.
    objective(x)
        Interpretable as the data's rotational inertia about the cluster
        centers. To be minimised.

    """

    def calculate_memberships(self, x):
        distances = self.distances(x)
        return np.arange(distances.shape[0])[:, np.newaxis] == np.argmin(
            distances, axis=0)

    def calculate_centers(self, x):
        return np.dot(self.memberships, x) / \
               np.sum(self.memberships, axis=1)[..., np.newaxis]

    def objective(self, x):
        distances = self.distances(x)
        return np.sum(self.memberships * distances)


class Fuzzy(CMeans):
    """Base class for fuzzy C-means clusters.

    Attributes
    ----------
    m : float
        Fuzziness parameter. Higher values reduce the rate of drop-off from
        full membership to zero membership.

    Methods
    -------
    fuzzifier(memberships)
        Fuzzification operator. By default, for memberships $u$ this is $u^m$.
    objective(x)
        Interpretable as the data's weighted rotational inertia about the
        cluster centers. To be minimised.

    """

    m = 2

    def fuzzifier(self, memberships):
        return np.power(memberships, self.m)

    def objective(self, x):
        distances = self.distances(x)
        return np.sum(self.fuzzifier(self.memberships) * distances)


class Probabilistic(Fuzzy):
    """Probabilistic C-means.

    In the probabilistic algorithm, sample points have total membership of
    unity, distributed equally among each of the centers. This tends to push
    cluster centers away from each other.

    Methods
    -------
    calculate_memberships(x)
        Memberships are calculated from the distance :math:`d_{ij}` between the
        sample :math:`j` and the cluster center :math:`i`.

        .. math::

            u_{ik} = \left(\sum_j \left(\frac{d_{ik}}{d_{jk}}\right)^\frac{2}{m
            -1} \right)^{-1}
    calculate_centers(x)
        New centers are calculated as the mean of the points closest to them,
        weighted by the fuzzified memberships.

        .. math::

            c_i = \left. \sum_k u_{ik}^m x_k \middle/ \sum_k u_{ik} \right.

    """
    def calculate_memberships(self, x):
        distances = self.distances(x)
        return 1 / np.sum(
            np.power(np.divide(distances, distances[:, np.newaxis]),
                     2 / (self.m - 1)), axis=0)

    def calculate_centers(self, x):
        return np.dot(self.fuzzifier(self.memberships), x) / \
               np.sum(self.fuzzifier(self.memberships), axis=1)[..., np.newaxis]


class Possibilistic(Fuzzy):
    """Possibilistic C-means.

    In the possibilistic algorithm, sample points are assigned memberships
    according to their relative proximity to the centers. This is controlled
    through a weighting to the cluster centers, approximately the variance of
    each cluster.

    Methods
    -------
    calculate_memberships(x)
        Memberships are calculated from the distance $d_{ij}$ between the sample
        $j$ and the cluster center $i$, and the weighting $w_i$ of each center.

        .. math::

            u_{ik} = \left(1 + \left(\frac{d_{ik}}{w_i}\right)^\frac{1}{m
            -1} \right)^{-1}
    calculate_centers(x)
        New centers are calculated as the mean of the points closest to them,
        weighted by the fuzzified memberships.

        .. math::

            c_i = \left. \sum_k u_{ik}^m x_k \middle/ \sum_k u_{ik} \right.

    """

    initialization = staticmethod(initialize_probabilistic)
    _weights = None

    def weights(self, x):
        if self._weights is None:
            distances = self.distances(x)
            memberships = self.calculate_memberships(x)
            self._weights = np.sum(self.fuzzifier(memberships) * distances,
                                   axis=1) / np.sum(self.fuzzifier(memberships),
                                                    axis=1)
        return self._weights

    def calculate_memberships(self, x):
        distances = self.distances(x)
        return (1. + (distances / self.weights(x)[:, np.newaxis]) ** (
            1. / (self.m - 1))) ** -1.

    def calculate_centers(self, x):
        return np.divide(np.dot(self.fuzzifier(self.memberships), x),
                         np.sum(self.fuzzifier(self.memberships), axis=1)[
                             ..., np.newaxis])


class GustafsonKesselMixin(Fuzzy):
    """Gives clusters ellipsoidal character.

    The Gustafson-Kessel algorithm redefines the distance measurement such that
    clusters may adopt ellipsoidal shapes. This is achieved through updates to
    a covariance matrix assigned to each cluster center.

    Examples
    --------
    Create a algorithm for probabilistic clustering with ellipsoidal clusters:

    >>> class ProbabilisticGustafsonKessel(GustafsonKesselMixin, Probabilistic):
    >>>     pass
    >>> pgk = ProbabilisticGustafsonKessel()
    >>> pgk.fit(x)

    """

    def distances(self, x):
        d = x - self.centers[:, np.newaxis]
        covariance = self.covariance(x, self.centers)
        left_multiplier = np.einsum('...ij,...jk', d, np.linalg.inv(covariance))
        return np.sum(left_multiplier * d, axis=2)

    def covariance(self, u, v):
        q, p = v.shape
        if self.memberships is None:
            return (np.eye(p)[..., np.newaxis] * np.ones((p, q))).T
        vector_difference = u - v[:, np.newaxis]
        fuzzy_memberships = self.fuzzifier(self.memberships)
        right_multiplier = np.einsum('...i,...j->...ij', vector_difference,
                                     vector_difference)
        einstein_sum = \
            np.einsum('...i,...ijk', fuzzy_memberships, right_multiplier) / \
            np.sum(fuzzy_memberships, axis=1)[..., np.newaxis, np.newaxis]
        return einstein_sum / np.power(np.linalg.det(einstein_sum), (1. / p))[
            ..., np.newaxis, np.newaxis]
