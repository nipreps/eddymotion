# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
"""Integration tests."""

import nibabel as nb
import nitransforms as nt
import numpy as np

from eddymotion.data.dmri import DWI
from eddymotion.estimator import EddyMotionEstimator


def test_proximity_estimator_trivial_model(datadir):
    """Check the proximity of transforms estimated by the estimator with a trivial B0 model."""

    dwdata = DWI.from_filename(datadir / "dwi.h5")
    b0nii = nb.Nifti1Image(dwdata.bzero, dwdata.affine, None)

    # Generate a list of large-yet-plausible bulk-head motion.
    xfms = nt.linear.LinearTransformsMapping(
        [
            nb.affines.from_matvec(nb.eulerangles.euler2mat(x=0.03, z=0.005), (0.8, 0.2, 0.2)),
            nb.affines.from_matvec(nb.eulerangles.euler2mat(x=0.02, z=0.005), (0.8, 0.2, 0.2)),
            nb.affines.from_matvec(nb.eulerangles.euler2mat(x=0.02, z=0.02), (0.4, 0.2, 0.2)),
            nb.affines.from_matvec(nb.eulerangles.euler2mat(x=-0.02, z=0.02), (0.4, 0.2, 0.2)),
            nb.affines.from_matvec(nb.eulerangles.euler2mat(x=-0.02, z=0.002), (0.0, 0.2, 0.2)),
            nb.affines.from_matvec(nb.eulerangles.euler2mat(y=-0.02, z=0.002), (0.0, 0.2, 0.2)),
            nb.affines.from_matvec(nb.eulerangles.euler2mat(y=-0.01, z=0.002), (0.0, 0.4, 0.2)),
        ],
        reference=b0nii,
    )

    # Induce motion into dataset (i.e., apply the inverse transforms)
    moved_nii = (~xfms).apply(b0nii, reference=b0nii)

    # Uncomment to see the moved dataset
    # moved_nii.to_filename(tmp_path / "test.nii.gz")
    # xfms.apply(moved_nii).to_filename(tmp_path / "ground_truth.nii.gz")

    # Wrap into dataset object
    dwi_motion = DWI(
        dataobj=moved_nii.dataobj,
        affine=b0nii.affine,
        bzero=dwdata.bzero,
        gradients=dwdata.gradients[..., : len(xfms)],
        brainmask=dwdata.brainmask,
    )

    estimator = EddyMotionEstimator()
    em_affines = estimator.estimate(
        dwdata=dwi_motion, models=("b0",), align_kwargs=None, seed=None
    )

    # Uncomment to see the realigned dataset
    # nt.linear.LinearTransformsMapping(
    #     em_affines,
    #     reference=b0nii,
    # ).apply(moved_nii).to_filename(tmp_path / "realigned.nii.gz")

    # For each moved b0 volume
    coords = xfms.reference.ndcoords.T
    for i, est in enumerate(em_affines):
        xfm = nt.linear.Affine(xfms.matrix[i], reference=b0nii)
        est = nt.linear.Affine(est, reference=b0nii)
        assert np.sqrt(((xfm.map(coords) - est.map(coords)) ** 2).sum(1)).mean() < 0.2
