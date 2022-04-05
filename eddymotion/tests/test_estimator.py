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
from pkg_resources import resource_filename as pkg_fn
import numpy as np
import pytest

import nitransforms as nt
from nibabel.eulerangles import euler2mat
from nibabel.affines import from_matvec
from nipype.interfaces.ants.registration import Registration
from eddymotion.dmri import DWI
from eddymotion.nifti import _to_nifti


@pytest.mark.parametrize("r_x", [0.0, 0.01, 0.1, 0.3])
@pytest.mark.parametrize("r_y", [0.0, 0.01, 0.1, 0.3])
@pytest.mark.parametrize("r_z", [0.0, 0.01, 0.1, 0.3])
@pytest.mark.parametrize("t_x", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("t_y", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("t_z", [0.0, 0.5, 1.0])
def test_ANTs_config_b0(pkg_datadir, tmpdir, r_x, r_y, r_z, t_x, t_y, t_z):
    """Check that the registration parameters for b=0
    gives a good estimate of known affine"""

    fixed = pkg_datadir / "b0.nii.gz"

    dwdata = DWI.from_filename((pkg_datadir, "/data/dwi.h5"))
    fixed = tmpdir / "b0.nii.gz"
    _to_nifti(dwdata.bzero, dwdata.affine, fixed)
    moving = tmpdir / "moving.nii.gz"
    tmpdir.chdir()
    T = from_matvec(euler2mat(x=r_x, y=r_y, z=r_z), (t_x, t_y, t_z))
    xfm = nt.linear.Affine(T, reference=fixed)

    (~xfm).apply(fixed).to_filename(moving)

    registration = Registration(
        terminal_output="file",
        from_file=pkg_fn(
            "eddymotion",
            "config/dwi-to-b0_level1.json",
        ),
        fixed_image=str(fixed.absolute()),
        moving_image=str(moving.absolute()),
    )
    result = registration.run(cwd=str(tmpdir)).outputs
    xform = nt.io.itk.ITKLinearTransform.from_filename(
        result.forward_transforms[0]
    ).to_ras(reference=fixed, moving=moving)

    assert np.all(
        abs(xfm.map(xfm.reference.ncoords) - xform.map(xfm.reference.ncoords)) < 0.1
    )
