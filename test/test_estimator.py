# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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
"""Unit tests exercising the estimator."""

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from nibabel.affines import from_matvec
from nibabel.eulerangles import euler2mat
from nipype.interfaces.ants.registration import Registration
from pkg_resources import resource_filename as pkg_fn

from eddymotion.data.dmri import DWI


@pytest.mark.parametrize("r_x", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("r_y", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("r_z", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("t_x", [0.0, 1.0])
@pytest.mark.parametrize("t_y", [0.0, 1.0])
@pytest.mark.parametrize("t_z", [0.0, 1.0])
def test_ANTs_config_b0(datadir, tmp_path, r_x, r_y, r_z, t_x, t_y, t_z):
    """Check that the registration parameters for b=0
    gives a good estimate of known affine"""

    fixed = tmp_path / "b0.nii.gz"
    moving = tmp_path / "moving.nii.gz"

    dwdata = DWI.from_filename(datadir / "dwi.h5")
    b0nii = nb.Nifti1Image(dwdata.bzero, dwdata.affine, None)
    b0nii.header.set_qform(dwdata.affine, code=1)
    b0nii.header.set_sform(dwdata.affine, code=1)
    b0nii.to_filename(fixed)

    T = from_matvec(euler2mat(x=r_x, y=r_y, z=r_z), (t_x, t_y, t_z))
    xfm = nt.linear.Affine(T, reference=b0nii)

    (~xfm).apply(b0nii, reference=b0nii).to_filename(moving)

    registration = Registration(
        terminal_output="file",
        from_file=pkg_fn(
            "eddymotion",
            "config/dwi-to-b0_level0.json",
        ),
        fixed_image=str(fixed.absolute()),
        moving_image=str(moving.absolute()),
        random_seed=1234,
    )

    result = registration.run(cwd=str(tmp_path)).outputs
    xform = nt.linear.Affine(
        nt.io.itk.ITKLinearTransform.from_filename(result.forward_transforms[0]).to_ras(),
        reference=b0nii,
    )

    coords = xfm.reference.ndcoords.T
    rms = np.sqrt(((xfm.map(coords) - xform.map(coords)) ** 2).sum(1)).mean()
    assert rms < 0.8
