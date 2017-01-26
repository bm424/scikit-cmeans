import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc


def contour(data, algorithm, axes=(0, 1), ax=None, resolution=200, threshold=0.5):
    """Scatter plot of data, with cluster contours overlaid.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        (n_samples, n_features)
        The original data.
    algorithm : :obj:`skcmeans.algorithms.CMeans`
        A cluster algorithm that has fitted the data.
    axes : :obj:`tuple` of :obj:`int`, optional
        Which of the data dimensions should be used for the 2-d plot. Default:
        `(0, 1)`
    ax : :obj:`matplotlib.axes.Axes`, optional
        Plot on an existing axis. The default behaviour is to create a new plot.
    resolution : int, optional
        The number of coordinates in both the x- and y-directions to use for the
        contours. Higher values take slightly longer to compute but lead to
        smoother contours. Default: 200
    threshold : float, optional
        Between 0 and 1. The cutoff point for the contours. Below this value,
        contours will not be plotted. Default: 0.5

    """
    if ax is None:
        ax = plt.figure().add_subplot(111)
    x, y = data[:, axes[0]], data[:, axes[1]]
    x_margin = 0.1 * x.ptp()
    y_margin = 0.1 * y.ptp()
    ax.scatter(x, y, c='k', s=4, linewidth=0)
    xv, yv = np.array(np.meshgrid(
        np.linspace(x.min() - x_margin, x.max() + x_margin, resolution),
        np.linspace(y.min() - y_margin, y.max() + y_margin, resolution)))
    shape = (data.shape[-1], 1, 1)
    data_means = np.tile(np.zeros_like(xv), shape) + data.mean(axis=0).reshape(
        shape)
    data_means[axes[0]] = xv
    data_means[axes[1]] = yv
    estimated_memberships = algorithm.calculate_memberships(
        data_means.reshape(data_means.shape[0], -1).T
            ).reshape(resolution, resolution, algorithm.n_clusters)
    estimated_memberships[estimated_memberships<threshold] = 0
    order = algorithm.centers[:, -1].argsort(axis=-1)
    color = plt.cm.viridis(np.linspace(0, 1, algorithm.n_clusters))
    for j, c in zip(range(algorithm.n_clusters), color):
        ax.contour(xv, yv, estimated_memberships[:, :, order[j]],
                   colors=mc.rgb2hex(c))
    if ax is plt:
        plt.xlim(x.min() - x_margin, x.max() + x_margin)
        plt.ylim(y.min() - y_margin, y.max() + y_margin)
    else:
        ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
        ax.set_ylim(y.min() - y_margin, y.max() + y_margin)