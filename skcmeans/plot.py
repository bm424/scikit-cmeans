import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc


def contour(data, algorithm, axes=(0, 1), ax=None, resolution=200,
            x_domain=None, y_domain=None, levels=0.5, labels=True,
            legend_loc="best", legend_order=None, legend_labels=None,
            color=None, size=4):
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
    ax, x, x_domain, y, y_domain = setup_plot(ax, axes, data, x_domain, y_domain)
    ax.scatter(x, y, c='k', s=size, linewidth=0)
    xv, yv = np.array(np.meshgrid(
        np.linspace(*x_domain, resolution),
        np.linspace(*y_domain, resolution)))
    shape = (data.shape[-1], 1, 1)
    data_means = np.tile(np.zeros_like(xv), shape) + data.mean(axis=0).reshape(
        shape)
    data_means[axes[0]] = xv
    data_means[axes[1]] = yv
    estimated_memberships = algorithm.calculate_memberships(
        data_means.reshape(data_means.shape[0], -1).T
            ).reshape(resolution, resolution, algorithm.n_clusters)
    if isinstance(levels, float):
        levels = np.arange(levels, 1.0, 0.1)
    order = algorithm.cluster_centers_[:, -1].argsort(axis=-1)
    if color is None:
        color = plt.cm.Vega20b(np.linspace(0, 1, algorithm.n_clusters))
    for j, c in zip(range(algorithm.n_clusters), color):
        print("Plotting cluster {} ({})".format(j, c))
        contours = ax.contour(xv, yv, estimated_memberships[:, :, order[j]],
                              colors=mc.rgb2hex(c), levels=levels)
        if labels:
            ax.clabel(contours, inline=1)
    if ax is plt:
        plt.xlim(*x_domain)
        plt.ylim(*y_domain)
    else:
        ax.set_xlim(*x_domain)
        ax.set_ylim(*y_domain)
    if legend_loc is not None:

        plt.legend(proxies, range(algorithm.n_clusters), title="Cluster number")


def scatter(data, algorithm, axes=(0, 1), ax=None, x_domain=None, y_domain=None, legend_loc="best", legend_order=None, legend_labels=None, color=None, size_multiplier=1.0):
    ax, x, x_domain, y, y_domain = setup_plot(ax, axes, data, x_domain, y_domain)
    order = algorithm.cluster_centers_[:, -1].argsort(axis=-1)
    if color is None:
        color = plt.cm.Vega20b(np.linspace(0, 1, algorithm.n_clusters))
    legend_handles = []
    for j, c in zip(range(algorithm.n_clusters), color):
        print("Plotting cluster {} ({})".format(j, c))
        condition = algorithm.memberships_[:, order[j]] > 0.5
        xv = x[condition]
        yv = y[condition]
        s = algorithm.memberships_[:, order[j]][condition] - 0.5
        handle = ax.scatter(xv, yv, s=s*size_multiplier, c=mc.rgb2hex(c))
        legend_handles.append(handle)
    if legend_loc is not None:
        plt.legend(legend_handles, range(algorithm.n_clusters), title="Cluster number")


def setup_plot(ax, axes, data, x_domain, y_domain):
    if ax is None:
        ax = plt.figure().add_subplot(111)
    x, y = data[:, axes[0]], data[:, axes[1]]
    if x_domain is None:
        x_margin = 0.1 * x.ptp()
        x_domain = x.min() - x_margin, x.max() + x_margin
    if y_domain is None:
        y_margin = 0.1 * y.ptp()
        y_domain = y.min() - y_margin, y.max() + y_margin
    return ax, x, x_domain, y, y_domain
