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
"""Eddymotion runner."""

from parser import parse_args
from eddymotion.data.dmri import DWI
from eddymotion.estimator import EddyMotionEstimator
import os


def main():
    """Entry point."""
    # Parse command-line arguments
    args = parse_args()

    # Open the data with the given file path
    dwi_dataset = DWI.from_filename(args.dwi_dir)

    # Initialize the EddyMotionEstimator
    estimator = EddyMotionEstimator()

    # Fit the EddyMotionEstimator to the DWI data
    estimated_affines = estimator.fit(
        dwi_dataset,
        align_kwargs=args.align_kwargs,
        models=args.models,
        omp_nthreads=args.omp_nthreads,
        n_jobs=args.n_jobs,
        seed=args.seed
    )

    if os.path.isdir(args.output):
        output_path = os.path.join(args.output, os.path.basename(args.dwi_dir))
    else:
        output_path = args.output

    dwi_dataset.to_filename(output_path)


if __name__ == "__main__":
    main()