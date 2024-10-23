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
from collections import defaultdict

# import nibabel as nib
import numpy as np
import pandas as pd
from dipy.core.gradients import gradient_table
from dipy.sims.voxel import single_tensor
from sklearn.model_selection import RepeatedKFold, cross_val_score

from eddymotion.model._sklearn import (
    EddyMotionGPR,
    SphericalKriging,
)
from eddymotion.testing import simulations as testsims


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

    # Use Scikit-learn cross validation
    scores = defaultdict(list, {})

    for n in args.kfold:
        for i in range(args.repeats):
            cv_scores = -1.0 * cross_validate(gtab, args.S0, args.evals1, evecs, args.snr, n)
            scores["rmse"] += cv_scores.tolist()
            scores["repeat"] += [i] * len(cv_scores)
            scores["n_folds"] += [n] * len(cv_scores)

        print(f"Finished {n}-fold cross-validation")

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv("cv_scores.tsv", sep="\t", index=None, na_rep="n/a")

    grouped = scores_df.groupby(["n_folds"])
    print(grouped[["rmse"]].mean())
    print(grouped[["rmse"]].std())

    # Plot - import plot_* from eddymotion.viz.signals
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
