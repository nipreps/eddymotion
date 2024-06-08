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

import nibabel as nb
import numpy as np
import pytest

from eddymotion.data.dmri import load


def _create_dwi_random_dataobj():
    rng = np.random.default_rng(1234)

    n_gradients = 10
    b0s = 1
    volumes = n_gradients + b0s
    b0_thres = 50
    bvals = np.hstack([b0s * [0], n_gradients * [1000]])
    bvecs = np.hstack([np.zeros((3, b0s)), rng.random((3, n_gradients))])

    vol_size = (34, 36, 24)

    dwi_dataobj = rng.random((*vol_size, volumes), dtype="float32")
    affine = np.eye(4, dtype="float32")
    brainmask_dataobj = rng.random(vol_size, dtype="float32")
    b0_dataobj = rng.random(vol_size, dtype="float32")
    gradients = np.vstack([bvecs, bvals[np.newaxis, :]], dtype="float32")
    fieldmap_dataobj = rng.random(vol_size, dtype="float32")

    return (
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        gradients,
        fieldmap_dataobj,
        b0_thres,
    )


def _create_dwi_random_data(
    dwi_dataobj,
    affine,
    brainmask_dataobj,
    b0_dataobj,
    fieldmap_dataobj,
):
    dwi = nb.Nifti1Image(dwi_dataobj, affine)
    brainmask = nb.Nifti1Image(brainmask_dataobj, affine)
    b0 = nb.Nifti1Image(b0_dataobj, affine)
    fieldmap = nb.Nifti1Image(fieldmap_dataobj, affine)

    return dwi, brainmask, b0, fieldmap


def _serialize_dwi_data(
    dwi,
    brainmask,
    b0,
    gradients,
    fieldmap,
    _tmp_path,
):
    dwi_fname = _tmp_path / "dwi.nii.gz"
    brainmask_fname = _tmp_path / "brainmask.nii.gz"
    b0_fname = _tmp_path / "b0.nii.gz"
    gradients_fname = _tmp_path / "gradients.txt"
    fieldmap_fname = _tmp_path / "fieldmap.nii.gz"

    nb.save(dwi, dwi_fname)
    nb.save(brainmask, brainmask_fname)
    nb.save(b0, b0_fname)
    np.savetxt(gradients_fname, gradients.T)
    nb.save(fieldmap, fieldmap_fname)

    return (
        dwi_fname,
        brainmask_fname,
        b0_fname,
        gradients_fname,
        fieldmap_fname,
    )


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


def test_equality_operator(tmp_path):
    # Create some random data
    (
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        gradients,
        fieldmap_dataobj,
        b0_thres,
    ) = _create_dwi_random_dataobj()

    dwi, brainmask, b0, fieldmap = _create_dwi_random_data(
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        fieldmap_dataobj,
    )

    (
        dwi_fname,
        brainmask_fname,
        b0_fname,
        gradients_fname,
        fieldmap_fname,
    ) = _serialize_dwi_data(
        dwi,
        brainmask,
        b0,
        gradients,
        fieldmap,
        tmp_path,
    )

    dwi_obj = load(
        dwi_fname,
        gradients_file=gradients_fname,
        b0_file=b0_fname,
        brainmask_file=brainmask_fname,
        fmap_file=fieldmap_fname,
        b0_thres=b0_thres,
    )
    hdf5_filename = tmp_path / "test_dwi.h5"
    dwi_obj.to_filename(hdf5_filename)

    round_trip_dwi_obj = load(hdf5_filename)

    # Symmetric equality
    assert dwi_obj == round_trip_dwi_obj
    assert round_trip_dwi_obj == dwi_obj
