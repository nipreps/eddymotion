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
Generate a synthetic dMRI signal and estimate values using Gaussian processes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dipy.core.sphere import Sphere

from eddymotion.model.gpr import EddyMotionGPR, SphericalKriging
from eddymotion.testing import simulations as testsims

SAMPLING_DIRECTIONS = 200


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
    parser.add_argument("--evals", help="Eigenvalues of the tensor", nargs="+", type=float)
    parser.add_argument("--snr", help="Signal to noise ratio", type=float)
    parser.add_argument(
        "dwi_gt_data_fname",
        help="Filename of NIfTI file containing the ground truth DWI signal",
        type=Path,
    )
    parser.add_argument(
        "bval_data_fname",
        help="Filename of b-val file containing the diffusion-weighting values",
        type=Path,
    )
    parser.add_argument(
        "bvec_data_fname",
        help="Filename of b-vecs file containing the diffusion-encoding gradient directions",
        type=Path,
    )
    parser.add_argument(
        "dwi_pred_data_fname",
        help="Filename of NIfTI file containing the predicted DWI signal",
        type=Path,
    )
    parser.add_argument(
        "bvec_pred_data_fname",
        help="Filename of b-vecs file containing the diffusion-encoding gradient b-vecs where "
        "the prediction is done",
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
    """Main function for running the experiment."""
    parser = _build_arg_parser()
    args = _parse_args(parser)

    seed = 1234
    n_voxels = 100

    data, gtab = testsims.simulate_voxels(
        args.S0,
        args.hsph_dirs,
        bval_shell=args.bval_shell,
        snr=args.snr,
        n_voxels=n_voxels,
        evals=args.evals,
        seed=seed,
    )

    # Save the generated signal and gradient table
    testsims.serialize_dmri(
        data, gtab, args.dwi_gt_data_fname, args.bval_data_fname, args.bvec_data_fname
    )

    # Fit the Gaussian Process regressor and predict on an arbitrary number of
    # directions
    a = 1.15
    lambda_s = 120
    alpha = 100
    gpr = EddyMotionGPR(
        kernel=SphericalKriging(a=a, lambda_s=lambda_s),
        alpha=alpha,
        optimizer=None,
    )

    # Use all available data to train the GP
    X_train = gtab[~gtab.b0s_mask].bvecs
    y = data[:, ~gtab.b0s_mask]

    gpr_fit = gpr.fit(X_train, y.T)

    # Predict on the testing data, plus a series of random directions
    theta, phi = testsims.create_random_polar_coordinates(SAMPLING_DIRECTIONS, seed=seed)
    sph = Sphere(theta=theta, phi=phi)

    X_test = np.vstack([gtab[~gtab.b0s_mask].bvecs, sph.vertices])

    predictions = gpr_fit.predict(X_test)

    # Save the predicted data
    testsims.serialize_dwi(predictions.T, args.dwi_pred_data_fname)
    np.savetxt(args.bvec_pred_data_fname, X_test.T, fmt="%.3f")


if __name__ == "__main__":
    main()
