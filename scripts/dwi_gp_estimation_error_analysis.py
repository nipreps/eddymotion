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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_score

from eddymotion.model._sklearn import (
    EddyMotionGPR,
    SphericalKriging,
)
from eddymotion.testing import simulations as testsims


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    cv: int,
    n_repeats: int,
    gpr: EddyMotionGPR,
) -> dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Perform the experiment by estimating the dMRI signal using a Gaussian process model.

    Parameters
    ----------
    X : :obj:`~numpy.ndarray`
        Diffusion-encoding gradient vectors.
    y : :obj:`~numpy.ndarray`
        DWI signal.
    cv : :obj:`int`
        Number of folds.
    n_repeats : :obj:`int`
        Number of times the cross-validator needs to be repeated.
    gpr : obj:`~eddymotion.model._sklearn.EddyMotionGPR`
        The eddymotion Gaussian process regressor object.

    Returns
    -------
    :obj:`dict`
        Data for the predicted signal and its error.

    """

    rkf = RepeatedKFold(n_splits=cv, n_repeats=n_repeats)
    scores = cross_val_score(gpr, X, y, scoring="neg_root_mean_squared_error", cv=rkf)
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
    parser.add_argument("bval_shell", help="Shell b-value", type=int)
    parser.add_argument("S0", help="S0 value", type=float)
    parser.add_argument(
        "error_data_fname",
        help="Filename of TSV file containing the data to plot",
        type=Path,
    )
    parser.add_argument("--evals", help="Eigenvalues of the tensor", nargs="+", type=float)
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
    """Main function for running the experiment."""
    parser = _build_arg_parser()
    args = _parse_args(parser)

    n_voxels = 100

    data, gtab = testsims.simulate_voxels(
        args.S0,
        args.hsph_dirs,
        bval_shell=args.bval_shell,
        snr=args.snr,
        n_voxels=n_voxels,
        evals=args.evals,
        seed=None,
    )

    X = gtab[~gtab.b0s_mask].bvecs
    y = data[:, ~gtab.b0s_mask]

    snr_str = args.snr if args.snr is not None else "None"

    a = 1.15
    lambda_s = 120
    alpha = 100
    gpr = EddyMotionGPR(
        kernel=SphericalKriging(a=a, lambda_s=lambda_s),
        alpha=alpha,
        optimizer=None,
    )

    # Use Scikit-learn cross validation
    scores = defaultdict(list, {})
    for n in args.kfold:
        for i in range(args.repeats):
            cv_scores = -1.0 * cross_validate(X, y.T, n, np.max(args.kfold) // n, gpr)
            scores["rmse"] += cv_scores.tolist()
            scores["repeat"] += [i] * len(cv_scores)
            scores["n_folds"] += [n] * len(cv_scores)
            scores["bval"] += [args.bval_shell] * len(cv_scores)
            scores["snr"] += [snr_str] * len(cv_scores)

        print(f"Finished {n}-fold cross-validation")

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(args.error_data_fname, sep="\t", index=None, na_rep="n/a")

    grouped = scores_df.groupby(["n_folds"])
    print(grouped[["rmse"]].mean())
    print(grouped[["rmse"]].std())


if __name__ == "__main__":
    main()
