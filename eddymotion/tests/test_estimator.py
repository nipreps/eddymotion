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
"""Unit tests exercising the estimator."""
from tempfile import TemporaryDirectory
from pkg_resources import resource_filename as pkg_fn
from pathlib import Path

import nitransforms as nt
import numpy as np
import random

from nibabel.eulerangles import euler2mat
from nibabel.affines import from_matvec
from nipype.interfaces.ants.registration import Registration


def test_ANTs_config_b0(pkg_datadir, tmpdir):
    """Check that the registration parameters for b=0
    gives a good estimate of known affine"""

    tmpdir.chdir()
    for i in range(100):
        # Generate test transfrom with random small parameters
        x = random.randint(0, 1000)
        y = random.randint(0, 1000)
        z = random.randint(0, 1000)
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        c = random.randint(0, 20)
        T = from_matvec(euler2mat(x=1 / x, y=1 / y, z=1 / z), [a / 5, b / 5, c / 5])
        xfm = nt.linear.Affine(T, reference=pkg_datadir / "b0.nii.gz")

        (~xfm).apply(pkg_datadir / "b0.nii.gz").to_filename(tmpdir / "moving.nii.gy")

        moving = tmpdir / "moving.nii.gz"
        fixed = pkg_datadir / "b0.nii.gz"
        registration = Registration(
            terminal_output="file",
            from_file=pkg_fn(
                "eddymotion",
                f"config/dwi-to-b0_level1.json",
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
