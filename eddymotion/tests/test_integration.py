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

import pytest  # noqa
import numpy as np
import nibabel as nb
from eddymotion.dmri import DWI
from eddymotion.estimator import EddyMotionEstimator
import nitransforms as nit


def test_proximity_estimator_trivial_model(pkg_datadir):
    """Check the proximity of the transforms estimated by :obj:`~eddymotion.estimator.EddyMotionEstimator` with a trivial B0 model."""
    _img = nb.load((pkg_datadir / 'b0.moving.nii.gz'))
    _moving_b0s_data = _img.get_fdata()[..., 1:]
    _b0 = _img.get_fdata()[..., 0]
    _affine = _img.affine.copy()
    _gradients = np.genfromtxt(
        fname=str(pkg_datadir / 'gradients.moving.tsv'),
        delimiter="\t",
        skip_header=0
    ).T
    _DWI = DWI(
        dataobj=_moving_b0s_data,
        affine=_affine,
        bzero=_b0,
        gradients=_gradients,
    )

    estimator = EddyMotionEstimator()
    em_affines = estimator.fit(
        dwdata=_DWI,
        n_iter=1,
        model="b0",
        align_kwargs=None,
        seed=None
    )

    # Load reference transforms originally applied
    ref_xfms = np.load((pkg_datadir / "b0.moving.transforms.npy"))

    # For each moved b0 volume
    for i, xfm in enumerate(em_affines):
        fixed_b0_img = nb.Nifti1Image(_b0, affine=_affine)
        xfm2 = nit.linear.Affine(
            ref_xfms[..., i],
            reference=fixed_b0_img
        )
        assert np.all(
            abs(xfm.map(xfm.reference.ndcoords.T) - xfm2.map(xfm.reference.ndcoords.T)) < 0.4
        )
