# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The NiPreps Developers <nipreps@gmail.com>
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

""" "
Simulate the DWI signal from a single fiber and plot the predicted signal using a Gaussian process
estimator.
"""

import argparse

import numpy as np
from dipy.core.geometry import sphere2cart
from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere, Sphere, disperse_charges
from dipy.sims.voxel import all_tensor_evecs, multi_tensor, single_tensor
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull, KDTree

from eddymotion.model._dipy import GaussianProcessModel

SAMPLING_DIRECTIONS = 200


def add_b0(bvals, bvecs):
    """Add a b0 signal to the diffusion-encoding gradient values and vectors."""

    _bvals = np.insert(bvals, 0, 0)
    _bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)

    return _bvals, _bvecs


def create_single_fiber_evecs(angles):
    """Create eigenvalues for a simulated single fiber."""

    sticks = np.array(sphere2cart(1, np.deg2rad(angles[0]), np.deg2rad(angles[1])))
    evecs = all_tensor_evecs(sticks)

    return evecs


def create_random_polar_coordinates(hsph_dirs, seed=1234):
    """Create random polar coordinate values"""

    rng = np.random.default_rng(seed)
    theta = np.pi * rng.random(hsph_dirs)
    phi = 2 * np.pi * rng.random(hsph_dirs)

    return theta, phi


def create_diffusion_encoding_gradient_dirs(hsph_dirs, iterations=5000, seed=1234):
    """Create the dMRI gradient-encoding directions."""

    # Create the gradient-encoding directions placing random points on a hemisphere
    theta, phi = create_random_polar_coordinates(hsph_dirs, seed=seed)
    hsph_initial = HemiSphere(theta=theta, phi=phi)

    # Move the points so that the electrostatic potential energy is minimized
    hsph_updated, potential = disperse_charges(hsph_initial, iterations)

    # Create a sphere
    return Sphere(xyz=np.vstack((hsph_updated.vertices, -hsph_updated.vertices)))


def create_single_shell_gradient_table(hsph_dirs, bval_shell, iterations=5000):
    """Create a single-shell gradient table."""

    # Create diffusion-encoding gradient directions
    sph = create_diffusion_encoding_gradient_dirs(hsph_dirs, iterations=iterations)

    # Create the gradient bvals and bvecs
    vertices = sph.vertices
    values = np.ones(vertices.shape[0])
    bvecs = vertices
    bvals = bval_shell * values

    # Add a b0 value to the gradient table
    bvals, bvecs = add_b0(bvals, bvecs)
    return gradient_table(bvals, bvecs)


def determine_fiber_count(evals):
    """Determine the fiber count."""

    evals_count = len(evals)

    # Create the DWI signal using a single tensor
    if evals_count == 3:
        return 1
    elif evals_count == 6:
        return 2
    elif evals_count == 9:
        return 3
    else:
        raise NotImplementedError(
            "Diffusion gradient-encoding signal generation not implemented for more than 3 fibers"
        )


def create_single_tensor_signal(angles, evals, S0, snr, rng, gtab):
    """Create a DWI signal with a single tensor."""

    # Create eigenvectors for a single fiber
    evecs = create_single_fiber_evecs(angles)

    return single_tensor(gtab, S0=S0, evals=evals, evecs=evecs, snr=snr, rng=rng)


def create_multi_tensor_signal(angles, evals, S0, snr, rng, gtab):
    """Create a DWI signal with multiple tensors."""

    # Signal fraction: percentage of the contribution of each tensor
    fractions = [100 / len(evals)] * len(evals)

    # signal, sticks = multi_tensor(
    #    gtab, evals, S0=S0, angles=angles, fractions=fractions, snr=snr, rng=rng
    # )
    # _evecs = np.array([all_tensor_evecs(_stick) for _stick in _sticks])
    signal, _ = multi_tensor(
        gtab, evals, S0=S0, angles=angles, fractions=fractions, snr=snr, rng=rng
    )

    return signal


def create_single_shell_signal(angles, gtab, S0, evals, snr):
    """Create a single-shell diffusion gradient-encoding signal."""

    # Fix the random number generator for reproducibility when generating the
    # signal
    seed = 1234
    rng = np.random.default_rng(seed)

    fiber_count = determine_fiber_count(evals)

    # Eigenvalues
    group_size = 3
    _evals = np.asarray([evals[i : i + group_size] for i in range(0, len(evals), group_size)])

    # Polar coordinates (theta, phi) of the principal axis of the tensor
    group_size = 2
    _angles = [tuple(angles[i : i + group_size]) for i in range(0, len(angles), group_size)]
    # Get the only in the lists for the single fiber case
    if fiber_count == 1:
        _evals = _evals[0]
        _angles = _angles[0]

    # Create the DWI signal using a single tensor
    if fiber_count == 1:
        signal = create_single_tensor_signal(_angles, _evals, S0, snr, rng, gtab)
    elif fiber_count == 2:
        signal = create_multi_tensor_signal(_angles, _evals, S0, snr, rng, gtab)
    elif fiber_count == 3:
        signal = create_multi_tensor_signal(_angles, _evals, S0, snr, rng, gtab)
    else:
        raise NotImplementedError(
            "Diffusion gradient-encoding signal generation not implemented for more than 3 fibers"
        )

    return signal


def get_query_vectors(gtab, train_mask):
    """Get the diffusion-encoding gradient vectors where the signal is to be estimated from the
    gradient table and the training mask: the vectors of interest are those that are masked in
    the training mask. b0 values are excluded."""

    idx = np.logical_and(~train_mask, ~gtab.b0s_mask)
    return gtab.bvecs[idx], np.where(idx)[0]


def create_random_train_mask(gtab, size, seed=1234):
    """Create a mask for the gradient table where a ``size`` number of indices will be
    excluded. b0 values are excluded."""

    rng = np.random.default_rng(seed)

    # Get the indices of the non-zero diffusion-encoding gradient vector indices
    nnzero_degv_idx = np.where(~gtab.b0s_mask)[0]

    if nnzero_degv_idx.size < size:
        raise ValueError(
            f"Requested {size} values for masking; gradient table has {nnzero_degv_idx.size} "
            "non-zero diffusion-encoding gradient vectors. Reduce the number of masked values."
        )

    lo = rng.choice(nnzero_degv_idx, size=size, replace=False)

    # Exclude the b0s
    zero_degv_idx = np.asarray(list(set(range(len(gtab.bvals))).difference(nnzero_degv_idx)))
    lo = np.hstack([zero_degv_idx, lo])

    train_mask = np.ones(len(gtab.bvals), dtype=bool)
    train_mask[lo] = False

    return train_mask


def perform_experiment(gtab, signal):
    """Perform experiment: estimate the dMRI signal on a set of directions fitting a
    Gaussian process to the rest of the data."""

    # Fix the random number generator for reproducibility when generating the
    # sampling directions
    # seed = 1234

    # Define the Gaussian process model parameters
    kernel_model = "spherical"
    lambda_s = 2.0
    a = 1.0
    sigma_sq = 0.5

    # Define the Gaussian process model instance
    gp_model = GaussianProcessModel(
        kernel_model=kernel_model, lambda_s=lambda_s, a=a, sigma_sq=sigma_sq
    )

    # Use all available data for training
    gpfit = gp_model.fit(signal[~gtab.b0s_mask], gtab[~gtab.b0s_mask])

    # Predict on an oversampled set of random directions over the unit sphere
    # theta, phi = create_random_polar_coordinates(SAMPLING_DIRECTIONS, seed=seed)
    # sph = Sphere(theta=theta, phi=phi)

    # ToDo
    # Not sure why all predictions are zero in gpfit.predict(sph.vertices)
    # Also, when creating the convex hull, the gtab required is the one that
    # would correspond to the new directions, so a new gtab would need to be
    # generated
    # return gpfit.predict(sph.vertices), sph.vertices
    # For now, predict on the same data
    return gpfit.predict(gtab[~gtab.b0s_mask].bvecs), gtab[~gtab.b0s_mask].bvecs


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


def check_fiber_data_args(angles, evals):
    """Check that the number of angle and eigenvalue elements to build a
    synthetic fiber signal are appropriate."""

    angles_count = len(angles)
    evals_count = len(evals)

    if angles_count % 2 != 0:
        raise ValueError(f"Two fiber angles required per fiber; {angles_count} provided")

    # Create the DWI signal using a single tensor
    if evals_count % 3 != 0:
        raise ValueError(
            f"Three fiber DTI model eigenvalues required per fiber; {evals_count} provided"
        )

    if len(angles) == 2 and len(evals) == 3:
        pass
    elif len(angles) == 4 and len(evals) == 6:
        pass
    elif len(angles) == 6 and len(evals) == 9:
        pass
    else:
        raise ValueError(
            "Fiber angle and fiber DTI model eigenvalue counts do not match; "
            f"{angles_count}, {evals_count} provided"
        )


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "hsph_dirs",
        help="Number of diffusion gradient-encoding directions in the half sphere",
        type=int,
    )
    parser.add_argument(
        "bval_shell",
        help="Shell b-value",
        type=float,
    )
    parser.add_argument(
        "S0",
        help="S0 value",
        type=float,
    )
    parser.add_argument(
        "--angles",
        help="Polar and azimuth angles of the tensor(s0",
        nargs="+",
        type=float,
    )
    parser.add_argument(
        "--evals",
        help="Eigenvalues of the tensor(s)",
        nargs="+",
        type=float,
    )
    parser.add_argument(
        "--snr",
        help="Signal to noise ratio",
        type=float,
    )
    return parser


def _parse_args(parser):
    args = parser.parse_args()

    return args


def main():
    parser = _build_arg_parser()
    args = _parse_args(parser)

    check_fiber_data_args(args.angles, args.evals)

    # Create a gradient table for a single-shell
    gtab = create_single_shell_gradient_table(args.hsph_dirs, args.bval_shell)

    # Create the DWI signal
    signal = create_single_shell_signal(args.angles, gtab, args.S0, args.evals, args.snr)

    # Estimate the dMRI signal using a Gaussian process estimator
    y_pred, y_pred_dirs = perform_experiment(gtab, signal)

    # Plot the predicted signal
    title = "GP model signal prediction"
    fig, _, _ = plot_prediction_surface(
        signal[~gtab.b0s_mask],
        y_pred,
        args.S0,
        gtab.bvecs[~gtab.b0s_mask],
        y_pred_dirs,
        title,
        "gray",
    )
    fig.savefig(args.gp_pred_plot_fname, format="svg")


if __name__ == "__main__":
    main()
