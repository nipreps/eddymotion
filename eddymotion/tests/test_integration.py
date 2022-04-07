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

import numpy as np
import nibabel as nb
from eddymotion.dmri import DWI
from eddymotion.estimator import EddyMotionEstimator
import nitransforms as nt


def test_proximity_estimator_trivial_model(pkg_datadir, tmp_path):
    """Check the proximity of transforms estimated by the estimator with a trivial B0 model."""

    dwdata = DWI.from_filename(pkg_datadir / "dwi.h5")
    b0nii = nb.Nifti1Image(dwdata.bzero, dwdata.affine, None)

    xfms = nt.linear.load(
        pkg_datadir / "head-motion-parameters.aff12.1D",
        fmt="afni",
    )
    xfms.reference = b0nii

    # Generate a dataset with 10 b-zeros and motion
    dwi_motion = DWI(
        dataobj=(~xfms).apply(b0nii, reference=b0nii).dataobj,
        affine=b0nii.affine,
        bzero=dwdata.bzero,
        gradients=dwdata.gradients[..., :10],
        brainmask=dwdata.brainmask,
    )

    estimator = EddyMotionEstimator()
    em_affines = estimator.fit(
        dwdata=dwi_motion,
        n_iter=1,
        model="b0",
        align_kwargs=None,
        seed=None
    )

    # For each moved b0 volume
    coords = xfms.reference.ndcoords.T
    for i, est in enumerate(em_affines):
        xfm = nt.linear.Affine(xfms.matrix[i], reference=b0nii)
        assert np.sqrt(
            ((xfm.map(coords) - est.map(coords))**2).sum(1)
        ).mean() < 0.2
