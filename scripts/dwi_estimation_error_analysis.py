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

from __future__ import annotations

import argparse

import numpy as np
from dipy.core.geometry import sphere2cart
from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere, Sphere, disperse_charges
from dipy.sims.voxel import all_tensor_evecs, single_tensor
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

from eddymotion.model._dipy import GaussianProcessModel


def add_b0(bvals: np.ndarray, bvecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert a b=0 at the front of the gradient table (b-values and b-vectors).

    Parameters
    ----------
    bvals : :obj:`~numpy.ndarray`
        Array of b-values.
    bvecs : :obj:`~numpy.ndarray`
        Array of b-vectors.

    Returns
    -------
    :obj:`tuple`
        Updated gradient table (b-values, b-vectors) including a b=0.

    """
    _bvals = np.insert(bvals, 0, 0)
    _bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)
    return _bvals, _bvecs


def create_single_fiber_evecs(theta: float = 0, phi: float = 0) -> np.ndarray:
    """
    Create eigenvectors for a simulated fiber given the polar coordinates of its pricipal axis.

    Parameters
    ----------
    theta : :obj:`float`
        Theta coordinate
    phi : :obj:`float`
        Phi coordinate

    Returns
    -------
    :obj:`~numpy.ndarray`
        Eigenvectors for a single fiber.

    """
    # Polar coordinates (theta, phi) of the principal axis of the tensor
    angles = np.array([theta, phi])
    sticks = np.array(sphere2cart(1, np.deg2rad(angles[0]), np.deg2rad(angles[1])))
    evecs = all_tensor_evecs(sticks)
    return evecs


def create_random_polar_coordinates(
    hsph_dirs: int, seed: int = 1234
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create random polar coordinates.

    Parameters
    ----------
    hsph_dirs : :obj:`int`
        Number of hemisphere directions.
    seed : :obj:`int`, optional
        Seed for the random number generator, by default 1234.

    Returns
    -------
    :obj:`tuple`
        Theta and Phi values of polar coordinates.

    """
    rng = np.random.default_rng(seed)
    theta = np.pi * rng.random(hsph_dirs)
    phi = 2 * np.pi * rng.random(hsph_dirs)
    return theta, phi


def create_diffusion_encoding_gradient_dirs(
    hsph_dirs: int, iterations: int = 5000, seed: int = 1234
) -> Sphere:
    """
    Create the dMRI gradient-encoding directions.

    Parameters
    ----------
    hsph_dirs : :obj:`int`
        Number of hemisphere directions.
    iterations : :obj:`int`, optional
        Number of iterations for charge dispersion, by default 5000.
    seed : :obj:`int`, optional
        Seed for the random number generator, by default 1234.

    Returns
    -------
    :obj:`~dipy.core.sphere.Sphere`
        A sphere with diffusion-encoding gradient directions.

    """
    # Create the gradient-encoding directions placing random points on a hemisphere
    theta, phi = create_random_polar_coordinates(hsph_dirs, seed=seed)
    hsph_initial = HemiSphere(theta=theta, phi=phi)

    # Move the points so that the electrostatic potential energy is minimized
    hsph_updated, _ = disperse_charges(hsph_initial, iterations)

    # Create a sphere
    return Sphere(xyz=np.vstack((hsph_updated.vertices, -hsph_updated.vertices)))


def create_single_shell_gradient_table(
    hsph_dirs: int, bval_shell: float, iterations: int = 5000
) -> gradient_table:
    """
    Create a single-shell gradient table.

    Parameters
    ----------
    hsph_dirs : :obj:`int`
        Number of hemisphere directions.
    bval_shell : :obj:`float`
        Shell b-value.
    iterations : :obj:`int`, optional
        Number of iterations for charge dispersion, by default 5000.

    Returns
    -------
    :obj:`~dipy.core.gradients.gradient_table`
        The gradient table for the single-shell.

    """
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


def get_query_vectors(
    gtab: gradient_table, train_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the diffusion-encoding gradient vectors for estimation from the gradient table.

    Get the diffusion-encoding gradient vectors where the signal is to be estimated from the
    gradient table and the training mask: the vectors of interest are those that are masked in
    the training mask. b0 values are excluded.

    Parameters
    ----------
    gtab : :obj:`~dipy.core.gradients.gradient_table`
        Gradient table.
    train_mask : :obj:`~numpy.ndarray`
        Mask for selecting training vectors.

    Returns
    -------
    :obj:`tuple`
        Gradient vectors and indices for estimation.

    """
    idx = np.logical_and(~train_mask, ~gtab.b0s_mask)
    return gtab.bvecs[idx], np.where(idx)[0]


def create_random_train_mask(gtab: gradient_table, size: int, seed: int = 1234) -> np.ndarray:
    """
    Create a random mask for the gradient table.

    Create a mask for the gradient table where a ``size`` number of indices will be
    excluded. b0 values are excluded.

    Parameters
    ----------
    gtab : :obj:`~dipy.core.gradients.gradient_table`
        Gradient table.
    size : :obj:`int`
        Number of indices to exclude.
    seed : :obj:`int`, optional
        Seed for the random number generator, by default 1234.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Training mask.

    Raises
    ------
    ValueError
        If the size of requested masked values is greater than available gradient vectors.

    """
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


def perform_experiment(
    gtab: gradient_table,
    S0: float,
    evals1: np.ndarray,
    evecs: np.ndarray,
    snr: float,
    repeats: int,
    kfold: list[int],
) -> dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Perform the experiment by estimating the dMRI signal using a Gaussian process model.

    Parameters
    ----------
    gtab : :obj:`~dipy.core.gradients.gradient_table`
        Gradient table.
    S0 : :obj:`float`
        S0 value.
    evals1 : :obj:`~numpy.ndarray`
        Eigenvalues of the tensor.
    evecs : :obj:`~numpy.ndarray`
        Eigenvectors of the tensor.
    snr : :obj:`float`
        Signal-to-noise ratio.
    repeats : :obj:`int`
        Number of repeats.
    kfold : :obj:`list`
        list of directions to leave out/predict.

    Returns
    -------
    :obj:`dict`
        Data for the predicted signal and its error.

    """

    # Fix the random number generator for reproducibility when generating the
    # signal
    rng = np.random.default_rng(1234)

    # Define the Gaussian process model parameter
    kernel_model = "spherical"
    lambda_s = 2.0
    a = 1.0
    sigma_sq = 0.5

    data = {}
    nzero_bvecs = gtab.bvecs[~gtab.b0s_mask]

    # Simulate the fitting a number of times: every time the signal created will be a little
    # different
    # for _ in range(repeats):
    # Create the DWI signal using a single tensor
    signal = single_tensor(gtab, S0=S0, evals=evals1, evecs=evecs, snr=snr, rng=rng)

    import pdb; pdb.set_trace()

    # Loop over the number of indices that are left out from the training/need to be predicted
    for n in kfold:
        # Assumptions:
        #  - Consecutive indices in the folds
        #  - A single b0
        kf = KFold(n_splits=n, shuffle=False)

        # Define the Gaussian process model instance
        gp_model = GaussianProcessModel(
            kernel_model=kernel_model, lambda_s=lambda_s, a=a, sigma_sq=sigma_sq
        )
        _data = []
        for _, (train_index, test_index) in enumerate(kf.split(nzero_bvecs)):
            # Create the training mask leaving out the requested number of samples
            # train_mask = create_random_train_mask(gtab, n)

            # Fit the Gaussian process
            # Add 1 to account for the b0
            gpfit = gp_model.fit(signal[train_index + 1], gtab[train_index + 1])

            # Predict the signal
            # X_qry, idx_qry = get_query_vectors(gtab, train_mask)
            # Add 1 to account for the b0
            idx_qry = test_index + 1
            X_qry = gtab[idx_qry].bvecs
            _y_pred, _y_std = gpfit.predict(X_qry)
            _data.append((idx_qry, signal[idx_qry], _y_pred, _y_std))
        data.update({n: _data})
    return data


def cross_validate(
    gtab: gradient_table,
    S0: float,
    evals1: np.ndarray,
    evecs: np.ndarray,
    snr: float,
    cv: int,
) -> dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Perform the experiment by estimating the dMRI signal using a Gaussian process model.

    Parameters
    ----------
    gtab : :obj:`~dipy.core.gradients.gradient_table`
        Gradient table.
    S0 : :obj:`float`
        S0 value.
    evals1 : :obj:`~numpy.ndarray`
        Eigenvalues of the tensor.
    evecs : :obj:`~numpy.ndarray`
        Eigenvectors of the tensor.
    snr : :obj:`float`
        Signal-to-noise ratio.
    cv : :obj:`int`
        number of folds

    Returns
    -------
    :obj:`dict`
        Data for the predicted signal and its error.

    """

    gp_params = {
        "kernel_model": "spherical",
        "lambda_s": 2.0,
        "a": 1.0,
        "sigma_sq": 0.5,
    }

    signal = single_tensor(gtab, S0=S0, evals=evals1, evecs=evecs, snr=snr)
    gpm = GaussianProcessModel(**gp_params)

    X = gtab[~gtab.b0s_mask].bvecs
    y = signal[~gtab.b0s_mask]

    scores = cross_val_score(gpm, X, y, scoring="neg_root_mean_squared_error", cv=cv)
    return scores


def compute_error(
    data: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
    repeats: int,
    kfolds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the error and standard deviation.

    Parameters
    ----------
    data : :obj:`dict`
        Data for the predicted signal and its error.
    repeats : :obj:`int`
        Number of repeats.
    kfolds : :obj:`dict`
        Number of k-folds.

    Returns
    -------
    :obj:`tuple`
        Mean RMSE and standard deviation.

    """
    mean_rmse = []
    std_dev = []

    # Loop over the range of indices that were predicted
    for vals in data.values():
        # repeats = 1
        # _data = np.array(vals[n * repeats : n * repeats + repeats])
        _signal = np.hstack([t[1] for t in vals])
        _pred = np.hstack([t[2] for t in vals])
        _rmse = root_mean_squared_error(_signal, _pred)
        # ToDo
        # Check here what is the value that we wil keep for the std
        _std = np.hstack([t[3] for t in vals])
        _std_dev = np.mean(_std)
        _std_dev = np.std(
            [root_mean_squared_error([v1], [v2]) for v1, v2 in zip(_signal, _pred, strict=False)]
        )  #  np.std(_rmse)
        mean_rmse.append(_rmse)
        std_dev.append(_std_dev)

    return np.asarray(mean_rmse), np.asarray(std_dev)


def plot_error(
    kfolds: list[int], mean: np.ndarray, std_dev: np.ndarray, xlabel: str, ylabel: str, title: str
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

    Returns
    -------
    :obj:`~matplotlib.pyplot.Figure`
        Matplotlib figure object.

    """
    fig, ax = plt.subplots()
    ax.plot(kfolds, mean, c="orange")
    ax.fill_between(kfolds, mean - std_dev, mean + std_dev, alpha=0.5, color="orange")
    ax.scatter(kfolds, mean, c="orange")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(kfolds)
    ax.set_xticklabels(kfolds)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for command-line interface.

    Returns
    -------
    :obj:`~argparse.ArgumentParser`
        Argument parser for the script.

    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "hsph_dirs",
        help="Number of diffusion gradient-encoding directions in the half sphere",
        type=int,
    )
    parser.add_argument("bval_shell", help="Shell b-value", type=float)
    parser.add_argument("S0", help="S0 value", type=float)
    parser.add_argument("--evals1", help="Eigenvalues of the tensor", nargs="+", type=float)
    parser.add_argument("--snr", help="Signal to noise ratio", type=float)
    parser.add_argument("--repeats", help="Number of repeats", type=int, default=5)
    parser.add_argument(
        "--kfold", help="Number of directions to leave out/predict", nargs="+", type=int
    )
    return parser


def _parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    parser : :obj:`~argparse.ArgumentParser`
        Argument parser for the script.

    Returns
    -------
    :obj:`~argparse.Namespace`
        Parsed arguments.
    """
    return parser.parse_args()


def main() -> None:
    """Main function for running the experiment and plotting the results."""
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
    scores = {
        n: cross_validate(gtab, args.S0, args.evals1, evecs, args.snr, n)
        for n in args.kfold for _ in range(args.repeats)
    }

    print({n: (np.mean(scores[n]), np.std(scores[n])) for n in args.kfold})

    # Compute the error
    # rmse, std_dev = compute_error(data, args.repeats, args.kfold)

    # Plot
    # xlabel = "N"
    # ylabel = "RMSE"
    # title = f"Gaussian process estimation\n(SNR={args.snr})"
    # _ = plot_error(args.kfold, rmse, std_dev, xlabel, ylabel, title)
    # fig = plot_error(args.kfold, rmse, std_dev)
    # fig.save(args.gp_pred_plot_error_fname, format="svg")


if __name__ == "__main__":
    main()
