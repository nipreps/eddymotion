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

"""
Simulate the DWI signal from a single fiber and analyze the prediction error of an estimator using
Gaussian processes.
"""

import argparse

import numpy as np
from dipy.core.geometry import sphere2cart
from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere, Sphere, disperse_charges
from dipy.sims.voxel import all_tensor_evecs, single_tensor
from matplotlib import pyplot as plt
from sklearn.metrics import root_mean_squared_error

from eddymotion.model._dipy import GaussianProcessModel


def add_b0(bvals, bvecs):
    """Add a b0 signal to the diffusion-encoding gradient values and vectors."""

    _bvals = np.insert(bvals, 0, 0)
    _bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)

    return _bvals, _bvecs


def create_single_fiber_evecs():
    """Create eigenvalues for a simulated single fiber."""

    # Polar coordinates (theta, phi) of the principal axis of the tensor
    angles = np.array([0, 0])
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
    hsph_updated, _ = disperse_charges(hsph_initial, iterations)

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


def perform_experiment(gtab, S0, evals1, evecs, snr, repeats, kfold):
    """Perform experiment: estimate the dMRI signal on a set of directions fitting a
    Gaussian process to the rest of the data."""

    # Fix the random number generator for reproducibility when generating the
    # signal
    rng = np.random.default_rng(1234)

    # Define the Gaussian process model parameters
    kernel_model = "spherical"
    lambda_s = 2.0
    a = 1.0
    sigma_sq = 0.5

    data = []

    # Loop over the number of indices that are left out from the training/need to be predicted
    for n in kfold:
        # Define the Gaussian process model instance
        gp_model = GaussianProcessModel(
            kernel_model=kernel_model, lambda_s=lambda_s, a=a, sigma_sq=sigma_sq
        )

        # Create the training mask leaving out the requested number of samples
        train_mask = create_random_train_mask(gtab, n)

        # Simulate the fitting a number of times: every time the signal created will be a little
        # different
        # for _ in range(repeats):
        # Create the DWI signal using a single tensor
        signal = single_tensor(gtab, S0=S0, evals=evals1, evecs=evecs, snr=snr, rng=rng)
        # Fit the Gaussian process
        gpfit = gp_model.fit(signal[train_mask], gtab[train_mask])

        # Predict the signal
        X_qry, idx_qry = get_query_vectors(gtab, train_mask)
        _y_pred, _y_std = gpfit.predict(X_qry)
        data.append((idx_qry, signal[idx_qry], _y_pred, _y_std))

    return data


def compute_error(data, repeats, kfolds):
    """Compute the error and standard deviation."""

    mean_rmse = []
    std_dev = []

    # Loop over the range of indices that were predicted
    for n in range(len(kfolds)):
        repeats = 1
        _data = np.array(data[n * repeats : n * repeats + repeats])
        _rmse = root_mean_squared_error(_data[0][1], _data[0][2])
        _std_dev = np.mean(_data[0][3])  #  np.std(_rmse)
        mean_rmse.append(_rmse)
        std_dev.append(_std_dev)

    return np.asarray(mean_rmse), np.asarray(std_dev)


def plot_error(kfolds, mean, std_dev):
    """Plot the error and standard deviation."""

    fig, ax = plt.subplots()
    ax.plot(kfolds, mean, c="orange")
    ax.fill_between(
        kfolds,
        mean - std_dev,
        mean + std_dev,
        alpha=0.5,
        color="orange",
    )
    ax.scatter(kfolds, mean, c="orange")
    ax.set_xlabel("N")
    ax.set_ylabel("RMSE")
    ax.set_xticks(kfolds)
    ax.set_xticklabels(kfolds)
    ax.set_title("Gaussian process estimation")
    fig.tight_layout()

    return fig


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
        "--evals1",
        help="Eigenvalues of the tensor",
        nargs="+",
        type=float,
    )
    parser.add_argument(
        "--snr",
        help="Signal to noise ratio",
        type=float,
    )
    parser.add_argument(
        "--repeats",
        help="Number of repeats",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--kfold",
        help="Number of directions to leave out/predict",
        nargs="+",
        type=int,
    )
    return parser


def _parse_args(parser):
    args = parser.parse_args()

    return args


def main():
    parser = _build_arg_parser()
    args = _parse_args(parser)

    # create eigenvectors for a single fiber
    evecs = create_single_fiber_evecs()

    # Create a gradient table for a single-shell
    gtab = create_single_shell_gradient_table(args.hsph_dirs, args.bval_shell)

    # Estimate the dMRI signal using a Gaussian process estimator
    # ToDo
    # Returning the index here is not useful as we cannot plot the signal as it
    # changes on every fold and we do not return it. Maybe we can set a random
    # value so that we return that one and we can plot it much like in the
    # notebook or maybe we leave that for a separate script/notebook ??
    data = perform_experiment(
        gtab, args.S0, args.evals1, evecs, args.snr, args.repeats, args.kfold
    )

    # Compute the error
    rmse, std_dev = compute_error(data, args.repeats, args.kfold)

    # Plot
    _ = plot_error(args.kfold, rmse, std_dev)
    # fig = plot_error(args.kfold, rmse, std_dev)
    # fig.save(args.gp_pred_plot_error_fname, format="svg")


if __name__ == "__main__":
    main()
