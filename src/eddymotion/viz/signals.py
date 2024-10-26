# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Visualizing signals and intermediate aspects of models."""

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull, KDTree
from scipy.stats import pearsonr


def plot_error(
    kfolds: list[int],
    mean: np.ndarray,
    std_dev: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    color: str = "orange",
    figsize: tuple[int, int] = (19.2, 10.8),
) -> plt.Figure:
    """
    Plot the error and standard deviation.

    Parameters
    ----------
    kfolds : :obj:`list`
        Number of k-folds.
    mean : :obj:`~numpy.ndarray`
        Mean RMSE values.
    std_dev : :obj:`~numpy.ndarray`
        Standard deviation values.
    xlabel : :obj:`str`
        X-axis label.
    ylabel : :obj:`str`
        Y-axis label.
    title : :obj:`str`
        Plot title.
    color : :obj:`str`, optional
        Plot color.
    figsize : :obj:`tuple`, optional
        Figure size.

    Returns
    -------
    :obj:`~matplotlib.pyplot.Figure`
        Matplotlib figure object.

    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(kfolds, mean, c=color)
    ax.fill_between(kfolds, mean - std_dev, mean + std_dev, alpha=0.5, color=color)
    ax.scatter(kfolds, mean, c=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(kfolds)
    ax.set_xticklabels(kfolds)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_estimation_carpet(gt_nii, gp_nii, gtab, suptitle, **kwargs):
    from nireports.reportlets.modality.dwi import nii_to_carpetplot_data
    from nireports.reportlets.nuisance import plot_carpet

    fig = plt.figure(layout="tight")
    gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
    fig.suptitle(suptitle)

    divide_by_b0 = False
    gt_data, segments = nii_to_carpetplot_data(gt_nii, bvals=gtab.bvals, divide_by_b0=divide_by_b0)

    title = "Ground truth"
    plot_carpet(gt_data, segments, subplot=gs[0, :], title=title, **kwargs)

    gp_data, segments = nii_to_carpetplot_data(gp_nii, bvals=gtab.bvals, divide_by_b0=divide_by_b0)

    title = "Estimated (GP)"
    plot_carpet(gt_data, segments, subplot=gs[1, :], title=title, **kwargs)

    return fig


def plot_correlation(x, y, title):
    r = pearsonr(x, y)

    # Fit a linear curve and estimate its y-values and their error
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * x + b
    y_err = x.std() * np.sqrt(1 / len(x) + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

    fig, ax = plt.subplots()
    ax.plot(x, y_est, "-", color="black", label=f"r = {r.correlation:.2f}")
    ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2, color="lightgray")
    ax.plot(x, y, marker="o", markersize="4", color="gray")

    ax.set_ylabel("Ground truth")
    ax.set_xlabel("Estimated")

    plt.title(title)
    plt.legend(loc="lower right")

    fig.tight_layout()

    return fig, r


def calculate_sphere_pts(points, center):
    """Calculate the location of each point when it is expanded out to the sphere."""

    kdtree = KDTree(points)  # tree of nearest points
    # d is an array of distances, i is an array of indices
    d, i = kdtree.query(center, points.shape[0])
    sphere_pts = np.zeros(points.shape, dtype=float)

    radius = np.amax(d)
    for p in range(points.shape[0]):
        sphere_pts[p] = points[i[p]] * radius / d[p]
    # points and the indices for where they were in the original lists
    return sphere_pts, i


def compute_dmri_convex_hull(s, dirs, mask=None):
    """Compute the convex hull of the dMRI signal s."""

    if mask is None:
        mask = np.ones(len(dirs), dtype=bool)

    # Scale the original sampling directions by the corresponding signal values
    scaled_bvecs = dirs[mask] * np.asarray(s)[:, np.newaxis]

    # Create the data for the convex hull: project the scaled vectors to a
    # sphere
    sphere_pts, sphere_idx = calculate_sphere_pts(scaled_bvecs, [0, 0, 0])

    # Create the convex hull: find the right ordering of vertices for the
    # triangles: ConvexHull finds the simplices of the points on the outside of
    # the data set
    hull = ConvexHull(sphere_pts)
    triang_idx = hull.simplices  # returns the list of indices for each triangle

    return scaled_bvecs, sphere_idx, triang_idx


def plot_surface(scaled_vecs, sphere_idx, triang_idx, title, cmap):
    """Plot a surface."""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter3D(
        scaled_vecs[:, 0], scaled_vecs[:, 1], scaled_vecs[:, 2], s=2, c="black", alpha=1.0
    )

    surface = ax.plot_trisurf(
        scaled_vecs[sphere_idx, 0],
        scaled_vecs[sphere_idx, 1],
        scaled_vecs[sphere_idx, 2],
        triangles=triang_idx,
        cmap=cmap,
        alpha=0.6,
    )

    ax.view_init(10, 45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)

    return fig, ax, surface


def plot_signal_data(y, ax):
    """Plot the data provided as a scatter plot"""

    ax.scatter(
        y[:, 0], y[:, 1], y[:, 2], color="red", marker="*", alpha=0.8, s=5, label="Original points"
    )


def plot_prediction_surface(y, y_pred, S0, y_dirs, y_pred_dirs, title, cmap):
    """Plot the prediction surface obtained by computing the convex hull of the
    predicted signal data, and plot the true data as a scatter plot."""

    # Scale the original sampling directions by the corresponding signal values
    y_bvecs = y_dirs * np.asarray(y)[:, np.newaxis]

    # Compute the convex hull
    y_pred_bvecs, sphere_idx, triang_idx = compute_dmri_convex_hull(y_pred, y_pred_dirs)

    # Plot the surface
    fig, ax, surface = plot_surface(y_pred_bvecs, sphere_idx, triang_idx, title, cmap)

    # Add the underlying signal to the plot
    # plot_signal_data(y_bvecs/S0, ax)
    plot_signal_data(y_bvecs, ax)

    fig.tight_layout()

    return fig, ax, surface
