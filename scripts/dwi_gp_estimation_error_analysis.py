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
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict, cross_val_score

from eddymotion.model.gpr import (
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
    gpr : obj:`~eddymotion.model.gpr.EddyMotionGPR`
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
        "--bval-shell",
        help="Shell b-value",
        type=float,
        default=1000,
    )
    parser.add_argument("--S0", help="S0 value", type=float, default=100)
    parser.add_argument(
        "--hsph-dirs",
        help="Number of diffusion gradient-encoding directions in the half sphere",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--output-scores",
        help="Filename of TSV file containing the data to plot",
        type=Path,
        default=Path() / "scores.tsv",
    )
    parser.add_argument(
        "-n",
        "--n-voxels",
        help="Number of diffusion gradient-encoding directions in the half sphere",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--write-inputs",
        help="Filename of NIfTI file containing the generated DWI signal",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-predicted",
        help="Filename of NIfTI file containing the predicted DWI signal",
        type=Path,
        default=None,
    )
    parser.add_argument("--evals", help="Eigenvalues of the tensor", nargs="+", type=float)
    parser.add_argument("--snr", help="Signal to noise ratio", type=float)
    parser.add_argument("--repeats", help="Number of repeats", type=int, default=5)
    parser.add_argument(
        "--kfold",
        help="Number of folds in repeated-k-fold cross-validation",
        nargs="+",
        type=int,
        default=None,
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

    data, gtab = testsims.simulate_voxels(
        args.S0,
        args.hsph_dirs,
        bval_shell=args.bval_shell,
        snr=args.snr,
        n_voxels=args.n_voxels,
        evals=args.evals,
        seed=None,
    )

    # Save the generated signal and gradient table
    if args.write_inputs:
        testsims.serialize_dmri(
            data,
            gtab,
            args.write_inputs,
            args.write_inputs.with_suffix(".bval"),
            args.write_inputs.with_suffix(".bvec"),
        )

    X = gtab[~gtab.b0s_mask].bvecs
    y = data[:, ~gtab.b0s_mask]

    snr_str = args.snr if args.snr is not None else "None"

    a = 1.15
    lambda_s = 120
    alpha = 1
    gpr = EddyMotionGPR(
        kernel=SphericalKriging(beta_a=a, beta_l=lambda_s),
        alpha=alpha,
        optimizer=None,
        # optimizer="Nelder-Mead",
        # disp=True,
        # ftol=1,
        # max_iter=2e5,
    )

    if args.kfold:
        # Use Scikit-learn cross validation
        scores = defaultdict(list, {})
        for n in args.kfold:
            for i in range(args.repeats):
                cv_scores = -1.0 * cross_validate(X, y.T, n, gpr)
                scores["rmse"] += cv_scores.tolist()
                scores["repeat"] += [i] * len(cv_scores)
                scores["n_folds"] += [n] * len(cv_scores)
                scores["bval"] += [args.bval_shell] * len(cv_scores)
                scores["snr"] += [snr_str] * len(cv_scores)

            print(f"Finished {n}-fold cross-validation")

        scores_df = pd.DataFrame(scores)
        scores_df.to_csv(args.output_scores, sep="\t", index=None, na_rep="n/a")

        grouped = scores_df.groupby(["n_folds"])
        print(grouped[["rmse"]].mean())
        print(grouped[["rmse"]].std())
    else:
        gpr.fit(X, y.T)
        print(gpr.kernel_)

    if args.output_predicted:
        cv = KFold(n_splits=3, shuffle=False, random_state=None)
        predictions = cross_val_predict(gpr, X, y.T, cv=cv)

        testsims.serialize_dwi(predictions.T, args.output_predicted)


if __name__ == "__main__":
    main()
