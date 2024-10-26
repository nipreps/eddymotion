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
Plot the RMSE (mean and std dev) and prediction surface from the predicted DWI
signal estimated using Gaussian processes k-fold cross-validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs

from eddymotion.viz.signals import plot_error, plot_prediction_surface


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
        "error_data_fname",
        help="Filename of TSV file containing the error data to plot",
        type=Path,
    )
    parser.add_argument(
        "dwi_gt_data_fname",
        help="Filename of NIfTI file containing the ground truth DWI signal",
        type=Path,
    )
    parser.add_argument(
        "bval_data_fname",
        help="Filename of b-val file containing the diffusion-encoding gradient b-vals",
        type=Path,
    )
    parser.add_argument(
        "bvec_data_fname",
        help="Filename of b-vecs file containing the diffusion-encoding gradient b-vecs",
        type=Path,
    )
    parser.add_argument(
        "dwi_pred_data_fname",
        help="Filename of NIfTI file containing the predicted DWI signal",
        type=Path,
    )
    parser.add_argument(
        "error_plot_fname",
        help="Filename of SVG file where the error plot will be saved",
        type=Path,
    )
    parser.add_argument(
        "signal_surface_plot_fname",
        help="Filename of SVG file where the predicted signal plot will be saved",
        type=Path,
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

    df = pd.read_csv(args.error_data_fname, sep="\t", keep_default_na=False, na_values="n/a")

    # Plot the prediction error
    kfolds = sorted(np.unique(df["n_folds"].values))
    snr = np.unique(df["snr"].values).item()
    rmse_data = [df.groupby("n_folds").get_group(k)["rmse"].values for k in kfolds]
    axis = 1
    mean = np.mean(rmse_data, axis=axis)
    std_dev = np.std(rmse_data, axis=axis)
    xlabel = "k"
    ylabel = "RMSE"
    title = f"Gaussian process estimation\n(SNR={snr})"
    fig = plot_error(kfolds, mean, std_dev, xlabel, ylabel, title)
    fig.savefig(args.error_plot_fname)
    plt.close(fig)

    # Plot the predicted DWI signal at a single voxel

    # Load the dMRI data
    signal = nib.load(args.dwi_gt_data_fname).get_fdata()
    y_pred = nib.load(args.dwi_pred_data_fname).get_fdata()

    bvals, bvecs = read_bvals_bvecs(str(args.bval_data_fname), str(args.bvec_data_fname))
    gtab = gradient_table(bvals, bvecs)

    # Pick one voxel randomly
    rng = np.random.default_rng(1234)
    idx = rng.integers(0, signal.shape[0], size=1).item()

    title = "GP model signal prediction"
    fig, _, _ = plot_prediction_surface(
        signal[idx, ~gtab.b0s_mask],
        y_pred[idx],
        signal[idx, gtab.b0s_mask].item(),
        gtab[~gtab.b0s_mask].bvecs,
        gtab[~gtab.b0s_mask].bvecs,
        title,
        "gray",
    )
    fig.savefig(args.signal_surface_plot_fname, format="svg")


if __name__ == "__main__":
    main()
