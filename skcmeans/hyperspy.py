from hyperspy.learn.mva import LearningResults
from skcmeans.algorithms import *


class ProbabilisticGK(Probabilistic, GustafsonKesselMixin):
    pass


def cluster(signal,
            n_clusters=2,
            algorithm='probabilistic',
            gustafson_kessel=False,
            use_decomposition_results=False,
            reproject=True,
            navigation_mask=None,
            signal_mask=None,
            **kwargs
            ):
    """Fuzzy c-means clustering with a choice of algorithms.

    Results are stored in `learning_results`.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to find.
    algorithm : 'hard' | 'probabilistic' | 'possibilistic'
        Algorithm used to find cluster centers and memberships. Refer to `
        scikit-cmeans` documentation for further information.
    gustafson_kessel : bool
        If True, the Gustafson-Kessel variant of the above algorithms will
        be used, allowing the clusters to have ellipsoidal character.
    use_decomposition_results : bool
        If True (recommended) the signal's decomposition results are used
        for clustering. If this option is not used, a `MemoryError` may
        arise during the clustering algorithm unless the Signal has a fairly
        small `signal_dimension`.
    reproject : bool
        If True and `use_decomposition_results` is True, the derived cluster
        centers will be reprojected into the original data dimension using
        the decomposition factors.
    navigation_mask : boolean numpy array
        The navigation locations marked as True are not used in the
        decomposition.
    signal_mask : boolean numpy array
        The signal locations marked as True are not used in the
        decomposition.
    **kwargs
        Additional parameters passed to the clustering algorithm. This may
        include `n_init`, the number of times the algorithm is restarted
        to optimize results.

    """
    if gustafson_kessel:
        algorithm += 'gk'
    algorithms = {
        'hard': Hard,
        'probabilistic': Probabilistic,
        'possibilistic': Possibilistic,
        'probabilisticgk': ProbabilisticGK,
    }
    Algorithm = algorithms[algorithm]
    if signal.axes_manager.navigation_size < 2:
        raise AttributeError("It is not possible to cluster a dataset "
                             "with navigation_size < 2")
    if not use_decomposition_results:
        signal._data_before_treatments = signal.data.copy()
    else:
        if signal.learning_results.loadings is not None:
            signal._data_before_treatments = signal.learning_results.loadings.copy()
        else:
            raise ValueError("`use_decomposition_results` is set to True but no "
                             "decomposition loadings have been found. Set "
                             "`use_decomposition_results` to False or run a "
                             "decomposition.")
    target = LearningResults()
    signal._unfolded4clustering = signal.unfold()

    try:
        # Deal with masks
        if hasattr(navigation_mask, 'ravel'):
            navigation_mask = navigation_mask.ravel()

        if hasattr(signal_mask, 'ravel'):
            signal_mask = signal_mask.ravel()
        if not use_decomposition_results:
            dc = signal.data if signal.axes_manager[0].index_in_array == 0 else signal.data.T
        else:
            dc = signal.learning_results.loadings.copy()
        if navigation_mask is None:
            navigation_mask = slice(None)
        else:
            navigation_mask = ~navigation_mask
        if signal_mask is None:
            signal_mask = slice(None)
        else:
            signal_mask = ~signal_mask

        # Cluster the masked data
        alg = Algorithm(n_clusters=n_clusters, **kwargs)
        alg.fit(dc[:, signal_mask][navigation_mask, :])
        memberships = alg.memberships_
        centers = alg.cluster_centers_

        target.memberships = memberships
        target.centers = centers
        target.n_clusters = n_clusters

        if signal._unfolded4clustering is True:
            folding = signal.metadata._HyperSpy.Folding
            target.original_shape = folding.original_shape

        # Reproject
        if use_decomposition_results and reproject:
            factors = signal.learning_results.factors
            centers = np.dot(centers, factors.T)
            target.centers = centers

        # Sort out masked data
        if not isinstance(signal_mask, slice):
            # Store the (inverted, as inputed) signal mask
            target.signal_mask = ~signal_mask.reshape(
                signal.axes_manager._signal_shape_in_array)
            if reproject not in ('both', 'signal'):
                centers = np.zeros(
                    (n_clusters, dc.shape[-1]))
                centers[:, signal_mask] = target.centers
                centers[:, ~signal_mask] = np.nan
                target.centers = centers
        if not isinstance(navigation_mask, slice):
            # Store the (inverted, as inputed) navigation mask
            target.navigation_mask = ~navigation_mask.reshape(
                signal.axes_manager._navigation_shape_in_array)
            if reproject not in ('both', 'navigation'):
                memberships = np.zeros(
                    (dc.shape[0], target.memberships.shape[1]))
                memberships[navigation_mask, :] = target.memberships
                memberships[~navigation_mask, :] = np.nan
                target.memberships = memberships
    finally:
        if signal._unfolded4clustering is True:
            signal.fold()
            signal._unfolded4clustering = False
        signal.learning_results.__dict__.update(target.__dict__)
        # undo any pre-treatments
        if not use_decomposition_results:
            signal.undo_treatments()
        return alg


def get_cluster_memberships(signal):
    """Return cluster memberships as a Signal.
    See Also
    --------
    get_cluster_centers
    """
    memberships = signal._get_loadings(
        signal.learning_results.memberships)
    memberships.axes_manager._axes[0].name = "Cluster index"
    memberships.metadata.General.title = \
        "Cluster memberships of " + signal.metadata.General.title
    return memberships


def get_cluster_centers(signal):
    """Return the cluster centers as a Signal.
    See Also
    -------
    get_cluster_memberships
    """
    centers = signal._get_factors(signal.learning_results.centers.T)
    centers.axes_manager._axes[0].name = "Cluster index"
    centers.metadata.General.title = ("Cluster centers of " +
                                     signal.metadata.General.title)
    return centers
