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
"""Unit tests exercising the dMRI data structure."""

import numpy as np
import pytest

from eddymotion.data.dmri import load


def test_load(datadir, tmp_path):
    """Check that the registration parameters for b=0
    gives a good estimate of known affine"""

    dwi_h5 = load(datadir / "dwi.h5")
    dwi_nifti_path = tmp_path / "dwi.nii.gz"
    gradients_path = tmp_path / "dwi.tsv"
    bvecs_path = tmp_path / "dwi.bvecs"
    bvals_path = tmp_path / "dwi.bvals"

    grad_table = np.hstack((np.zeros((4, 1)), dwi_h5.gradients))

    dwi_h5.to_nifti(dwi_nifti_path, insert_b0=True)
    np.savetxt(str(gradients_path), grad_table.T)
    np.savetxt(str(bvecs_path), grad_table[:3])
    np.savetxt(str(bvals_path), grad_table[-1])

    with pytest.raises(RuntimeError):
        load(dwi_nifti_path)

    # Try loading NIfTI + gradients table
    dwi_from_nifti1 = load(dwi_nifti_path, gradients_file=gradients_path)

    assert np.allclose(dwi_h5.dataobj, dwi_from_nifti1.dataobj)
    assert np.allclose(dwi_h5.bzero, dwi_from_nifti1.bzero)
    assert np.allclose(dwi_h5.gradients, dwi_from_nifti1.gradients)

    # Try loading NIfTI + b-vecs/vals
    dwi_from_nifti2 = load(
        dwi_nifti_path,
        bvec_file=bvecs_path,
        bval_file=bvals_path,
    )

    assert np.allclose(dwi_h5.dataobj, dwi_from_nifti2.dataobj)
    assert np.allclose(dwi_h5.bzero, dwi_from_nifti2.bzero)
    assert np.allclose(dwi_h5.gradients, dwi_from_nifti2.gradients)
