import numpy as np
from scipy.spatial.distance import cdist
from .initialization import initialize_random, initialize_probabilistic


class CMeans:

    metric = 'euclidean'
    initialization = staticmethod(initialize_random)

    def __init__(self, n_clusters=2, n_init=10, max_iter=300, tol=1e-4,
                 verbose=0, random_state=None, **kwargs):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.params = kwargs
        self.centers = None
        self.memberships = None

    def distances(self, x):
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
        j_new = self.objective(x)
        for i in range(self.max_iter):
            j_old = j_new
            self.update(x)
            j_new = self.objective(x)
            if j_old - j_new < self.tol:
                break

    def update(self, x):
        if self.centers is not None:
            self.centers = self.calculate_centers(x)
        else:
            self.centers = self.initialization(x, self.n_clusters,
                                               self.random_state)
        self.memberships = self.calculate_memberships(x)


class Hard(CMeans):

    def calculate_memberships(self, x):
        distances = self.distances(x)
        return np.arange(distances.shape[0])[:, np.newaxis] == np.argmin(
            distances, axis=0)

    def calculate_centers(self, x):
        return np.dot(self.memberships, x) /\
               np.sum(self.memberships, axis=1)[..., np.newaxis]

    def objective(self, x):
        distances = self.distances(x)
        return np.sum(self.memberships * distances)


class Fuzzy(CMeans):

    m = 2

    def fuzzifier(self, memberships):
        return np.power(memberships, self.m)

    def objective(self, x):
        distances = self.distances(x)
        return np.sum(self.fuzzifier(self.memberships) * distances)


class Probabilistic(Fuzzy):

    def calculate_memberships(self, x):
        distances = self.distances(x)
        return 1 / np.sum(
            np.power(np.divide(distances, distances[:, np.newaxis]),
                     2 / (self.m - 1)), axis=0)

    def calculate_centers(self, x):
        return np.dot(self.fuzzifier(self.memberships), x) / \
               np.sum(self.fuzzifier(self.memberships), axis=1)[..., np.newaxis]


class Possibilistic(Fuzzy):

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
        einstein_sum = np.einsum('...i,...ijk', fuzzy_memberships,
                                 right_multiplier) / \
                       np.sum(fuzzy_memberships, axis=1)[
                           ..., np.newaxis, np.newaxis]
        return einstein_sum / np.power(np.linalg.det(einstein_sum), (1. / p))[
            ..., np.newaxis, np.newaxis]