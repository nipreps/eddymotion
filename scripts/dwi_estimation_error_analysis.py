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

"""
Simulate the DWI signal from a single fiber and analyze the prediction error of an estimator using
Gaussian processes.
"""

from __future__ import annotations

import argparse

import matplotlib.gridspec as gridspec

# import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.sims.voxel import single_tensor
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score

from eddymotion.model._sklearn import (
    EddyMotionGPR,
    SphericalKriging,
)
from eddymotion.testing import simulations as testsims


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

    import pdb

    pdb.set_trace()

    # Loop over the number of indices that are left out from the training/need to be predicted
    for n in kfold:
        # Assumptions:
        #  - Consecutive indices in the folds
        #  - A single b0
        kf = KFold(n_splits=n, shuffle=False)

        # Define the Gaussian process model instance
        gp_model = EddyMotionGPR(
            kernel=SphericalKriging(a=a, lambda_s=lambda_s),
            alpha=sigma_sq,
            optimizer=None,
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

    signal = single_tensor(gtab, S0=S0, evals=evals1, evecs=evecs, snr=snr)
    gpm = EddyMotionGPR(
        kernel=SphericalKriging(a=2.15, lambda_s=120),
        alpha=50,
        optimizer=None,
    )

    X = gtab[~gtab.b0s_mask].bvecs
    y = signal[~gtab.b0s_mask]

    rkf = RepeatedKFold(n_splits=cv, n_repeats=120 // cv)
    scores = cross_val_score(gpm, X, y, scoring="neg_root_mean_squared_error", cv=rkf)
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
    evecs = testsims.create_single_fiber_evecs()

    # Create a gradient table for a single-shell
    gtab = testsims.create_single_shell_gradient_table(args.hsph_dirs, args.bval_shell)

    # Estimate the dMRI signal using a Gaussian process estimator
    # ToDo
    # Returning the index here is not useful as we cannot plot the signal as it
    # changes on every fold and we do not return it. Maybe we can set a random
    # value so that we return that one and we can plot it much like in the
    # notebook or maybe we leave that for a separate script/notebook ??
    scores = {
        n: cross_validate(gtab, args.S0, args.evals1, evecs, args.snr, n)
        for n in args.kfold
        for _ in range(args.repeats)
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

    # dirname = Path(args.gp_pred_plot_error_fname).parent
    # for key, val in data.items():
    #     # Recompose the DWI signal from the folds
    #     _signal = np.hstack([t[1] for t in val])
    #     _pred = np.hstack([t[2] for t in val])

    #     # Build the NIfTI images for the carpet plots
    #     gt_img = _signal[np.newaxis, np.newaxis, np.newaxis, :]
    #     gp_img = _pred[np.newaxis, np.newaxis, np.newaxis, :]
    #     affine = np.eye(4)
    #     gt_nii = nib.Nifti1Image(gt_img, affine)
    #     gp_nii = nib.Nifti1Image(gp_img, affine)

    #     title = f"DWI signal carpet plot\n(SNR={args.snr}; N={key})"
    #     _ = plot_estimation_carpet(gt_nii, gp_nii, gtab[~gtab.b0s_mask], title)
    #     # fname = dirname / f"carpet_plot_fold-{key}.svg"
    #     # fig.savefig(fname, format="svg")

    #     title = f"DWI signal correlation\n(SNR={args.snr}; N={key})"
    #     _ = plot_correlation(_signal, _pred, title)
    #     # fname = dirname / f"correlation_plot_fold-{key}.svg"
    #     # fig.savefig(fname, format="svg")


if __name__ == "__main__":
    main()
