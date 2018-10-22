"""Implementations of a number of C-means algorithms.

References
----------
.. [1] J. C. Bezdek, J. Keller, R. Krisnapuram, and N. R. Pal, Fuzzy models
   and algorithms for pattern recognition and image processing. Kluwer Academic
   Publishers, 2005.

"""
import numpy as np
from scipy.spatial.distance import cdist

from skcmeans import plot
from .initialization import initialize_random, initialize_probabilistic


class CMeans:
    """Base class for C-means algorithms.

    Parameters
    ----------
    metric : :obj:`string` or :obj:`function`
        The distance metric used. May be any of the strings specified for
        :obj:`cdist`, or a user-specified function.
    initialization : function
        The method used to initialize the cluster_centers_.

    Attributes
    ----------
    cluster_centers_ : :obj:`np.ndarray`
        (n_clusters, n_features)
        The derived or supplied cluster centers.
    memberships_ : :obj:`np.ndarray`
        (n_samples, n_clusters)
        The derived or supplied cluster memberships.

    """

    def __init__(self, n_clusters=2, n_init=10, max_iter=300, tol=1e-4,
                 verbosity=0, random_state=None, eps=1e-18,
                 metric='euclidean', initialization=initialize_random):
        """
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
        random_state : :obj:`int` or :obj:`np.random.RandomState`, optional
            The generator used for initialization. Using an integer fixes the seed.
        eps : float, optional
            To avoid numerical errors, zeros are sometimes replaced with a very
            small number, specified here.
        """
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbosity = verbosity
        self.random_state = random_state
        self.eps = eps
        self.metric = metric
        self.initialization = initialization
        self.cluster_centers_ = None
        self.memberships_ = None

    def distances(self, x):
        """Calculates the distance between data x and the cluster centers.

        The distance, by default, is calculated according to `metric`, but this
        method should be overridden by subclasses if required.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            Sample data.

        Returns
        -------
        :obj:`np.ndarray`
            (n_samples, n_clusters)
            Each entry (i, j) is the distance between sample i and cluster
            center j.

        """
        return cdist(x, self.cluster_centers_, metric=self.metric)

    def calculate_memberships(self, x):
        raise NotImplementedError(
            "`calculate_memberships` should be implemented by subclasses.")

    def calculate_centers(self, x):
        raise NotImplementedError(
            "`calculate_centers` should be implemented by subclasses.")

    def predict(self, x):
        """Predict the cluster labels.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            Sample data.

        """
        memberships = self.calculate_memberships(x)
        return np.argmax(memberships, axis=1)

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
        iterator = range(self.n_init)
        for i in iterator:
            self.cluster_centers_ = None
            self.memberships_ = None
            results = self.converge(x)
            objective = results['objective']
            if objective < objective_best:
                memberships_best = self.memberships_.copy()
                centers_best = self.cluster_centers_.copy()
                objective_best = objective
        self.memberships_ = memberships_best
        self.cluster_centers_ = centers_best
        return self

    def converge(self, x):
        """Finds `cluster_centers_` through an alternating optimization routine.

        Terminates when either the number of cycles reaches `max_iter` or the
        objective function changes by less than `tol`.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        """
        j_new = np.infty
        for i in range(self.max_iter):
            j_old = j_new
            self.update(x)
            j_new = self.objective(x)
            if np.abs(j_old - j_new) < self.tol:
                break
        results = {
            'memberships': self.memberships_,
            'centers': self.cluster_centers_,
            'objective': j_new,
            'objective_delta': abs(j_new - j_old),
            'n_iter': i + 1,
            'algorithm': self,
        }
        return results

    def update(self, x):
        """Updates cluster memberships_ and cluster_centers_ in a single cycle.

        If the cluster cluster_centers_ have not already been initialized, they are
        chosen according to `initialization`.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        """
        if not self._initialized:
            self.initialize(x)
        self.memberships_ = self.calculate_memberships(x)
        self.cluster_centers_ = self.calculate_centers(x)

    @property
    def _initialized(self):
        return self.cluster_centers_ is not None \
               and self.memberships_ is not None

    def initialize(self, x):
        x = self._check_fit_data(x)
        initialization = self.initialization(x, self.n_clusters, self.random_state)
        if self.cluster_centers_ is None and self.memberships_ is None:
            self.memberships_, self.cluster_centers_ = initialization
        elif self.memberships_ is None:
            self.memberships_ = initialization[0]
        elif self.cluster_centers_ is None:
            self.cluster_centers_ = initialization[1]

    def plot(self, x, method="contour", *args, **kwargs):
        if method is "contour":
            plot.contour(x, self, *args, **kwargs)
        elif method is "scatter":
            plot.scatter(x, self, *args, **kwargs)
        else:
            raise NotImplementedError("Method '{}' is not implemented.".format(method))

    def _check_fit_data(self, x):
        if x.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                x.shape[0], self.n_clusters))
        return x


class Hard(CMeans):
    """Hard C-means, equivalent to K-means clustering.

    Methods
    -------
    calculate_memberships(x)
        The membership of a sample is 1 to the closest cluster and 0 otherwise.
    calculate_centers(x)
        New cluster_centers_ are calculated as the mean of the points closest to them.
    objective(x)
        Interpretable as the data's rotational inertia about the cluster
        cluster_centers_. To be minimised.

    """

    def calculate_memberships(self, x):
        distances = self.distances(x)
        return (np.arange(distances.shape[1])[:, np.newaxis] == np.argmin(
            distances, axis=1)).T.astype("float")

    def calculate_centers(self, x):
        return np.dot(self.memberships_.T, x) / \
               np.sum(self.memberships_, axis=0)[..., np.newaxis]

    def objective(self, x):
        if self.memberships_ is None or self.cluster_centers_ is None:
            return np.infty
        distances = self.distances(x)
        return np.sum(self.memberships_ * distances)


class Fuzzy(CMeans):
    """Base class for fuzzy C-means clusters.

    Attributes
    ----------
    m : float
        Fuzziness parameter. Higher values reduce the rate of drop-off from
        full membership to zero membership.

    Methods
    -------
    fuzzifier(memberships_)
        Fuzzification operator. By default, for memberships_ $u$ this is $u^m$.
    objective(x)
        Interpretable as the data's weighted rotational inertia about the
        cluster cluster_centers_. To be minimised.

    """

    m = 2

    def __init__(self, *args, **kwargs):
        super(Fuzzy, self).__init__(*args, **kwargs)
        if 'm' in kwargs:
            self.m = kwargs['m']

    def fuzzifier(self, memberships):
        return np.power(memberships, self.m)

    def objective(self, x):
        if self.memberships_ is None or self.cluster_centers_ is None:
            return np.infty
        distances = self.distances(x)
        return np.sum(self.fuzzifier(self.memberships_) * distances)


class Probabilistic(Fuzzy):
    """Probabilistic C-means.

    In the probabilistic algorithm, sample points have total membership of
    unity, distributed equally among each of the cluster_centers_. This tends to push
    cluster cluster_centers_ away from each other.

    Methods
    -------
    calculate_memberships(x)
        Memberships are calculated from the distance :math:`d_{ij}` between the
        sample :math:`j` and the cluster center :math:`i`.

        .. math::

            u_{ik} = \left(\sum_j \left( \\frac{d_{ik}}{d_{jk}} \\right)^{\\frac{2}{m - 1}} \\right)^{-1}

    calculate_centers(x)
        New cluster_centers_ are calculated as the mean of the points closest to them,
        weighted by the fuzzified memberships_.

        .. math:: c_i = \left. \sum_k u_{ik}^m x_k \middle/ \sum_k u_{ik} \\right.

    """
    def calculate_memberships(self, x):
        distances = self.distances(x)
        distances[distances == 0.] = 1e-18
        return np.sum(np.power(
            np.divide(distances[:, :, np.newaxis], distances[:, np.newaxis, :]),
            2 / (self.m - 1)), axis=2) ** -1

    def calculate_centers(self, x):
        return np.dot(self.fuzzifier(self.memberships_).T, x) / \
               np.sum(self.fuzzifier(self.memberships_).T, axis=1)[..., np.newaxis]


class Possibilistic(Fuzzy):
    """Possibilistic C-means.

    In the possibilistic algorithm, sample points are assigned memberships_
    according to their relative proximity to the cluster_centers_. This is controlled
    through a weighting to the cluster cluster_centers_, approximately the variance of
    each cluster.

    Methods
    -------
    calculate_memberships(x)
        Memberships are calculated from the distance :math:`d_{ij}` between the
        sample :math:`j` and the cluster center :math:`i`, and the weighting
        :math:`w_i` of each center.

        .. math::

            u_{ik} = \left(1 + \left(\\frac{d_{ik}}{w_i}\\right)^\\frac{1}{m
            -1} \\right)^{-1}

    calculate_centers(x)
        New cluster_centers_ are calculated as the mean of the points closest to them,
        weighted by the fuzzified memberships_.

        .. math::

            c_i = \left. \sum_k u_{ik}^m x_k \middle/ \sum_k u_{ik} \\right.

    """

    _weights = None

    def __init__(self, n_clusters=2, n_init=10, max_iter=300, tol=1e-4,
                 verbosity=0, random_state=None, eps=1e-18,
                 metric='euclidean', initialization=initialize_probabilistic):
        super(Possibilistic, self).__init__(
            n_clusters, n_init, max_iter, tol, verbosity, random_state, eps,
            metric, initialization
        )

    def weights(self, x):
        if self._weights is None:
            distances = self.distances(x)
            memberships = self.memberships_
            self._weights = np.sum(self.fuzzifier(memberships) * distances,
                                   axis=0) / np.sum(self.fuzzifier(memberships),
                                                    axis=0)
        return self._weights

    def calculate_memberships(self, x):
        distances = self.distances(x)
        return (1. + (distances / self.weights(x)) ** (
            1. / (self.m - 1))) ** -1.

    def calculate_centers(self, x):
        return np.divide(np.dot(self.fuzzifier(self.memberships_).T, x),
                         np.sum(self.fuzzifier(self.memberships_), axis=0)[
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
    covariance = None

    def fit(self, x):
        """Optimizes cluster cluster_centers_ by restarting convergence several times.

        Extends the default behaviour by recalculating the covariance matrix
        with resultant memberships_ and cluster_centers_.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        """
        j_list = super(GustafsonKesselMixin, self).fit(x)
        self.covariance = self.calculate_covariance(x)
        return j_list

    def update(self, x):
        """Single update of the cluster algorithm.

        Extends the default behaviour by including a covariance calculation
        after updating the cluster_centers_

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        """
        self.initialize(x)
        self.cluster_centers_ = self.calculate_centers(x)
        self.covariance = self.calculate_covariance(x)
        self.memberships_ = self.calculate_memberships(x)

    def distances(self, x):
        covariance = self.covariance if self.covariance is not None \
            else self.calculate_covariance(x)
        d = x - self.cluster_centers_[:, np.newaxis]
        left_multiplier = \
            np.einsum('...ij,...jk', d, np.linalg.inv(covariance))
        return np.sum(left_multiplier * d, axis=2).T

    def calculate_covariance(self, x):
        """Calculates the covariance of the data `u` with cluster cluster_centers_ `v`.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            (n_samples, n_features)
            The original data.

        Returns
        -------
        :obj:`np.ndarray`
            (n_clusters, n_features, n_features)
            The covariance matrix of each cluster.

        """
        v = self.cluster_centers_
        if v is None:
            return None
        q, p = v.shape
        if self.memberships_ is None:
            # If no memberships_ have been calculated assume n-spherical clusters
            return (np.eye(p)[..., np.newaxis] * np.ones((p, q))).T
        q, p = v.shape
        vector_difference = x - v[:, np.newaxis]
        fuzzy_memberships = self.fuzzifier(self.memberships_)
        right_multiplier = \
            np.einsum('...i,...j->...ij', vector_difference, vector_difference)
        einstein_sum = \
            np.einsum('i...,...ijk', fuzzy_memberships, right_multiplier) / \
            np.sum(fuzzy_memberships, axis=0)[..., np.newaxis, np.newaxis]
        return np.nan_to_num(
            einstein_sum / (np.linalg.det(einstein_sum) ** (1 / q))[
                ..., np.newaxis, np.newaxis])
