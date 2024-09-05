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
"""Unit tests exercising data filtering utilities."""
import nibabel as nb
import numpy as np

import pytest

from eddymotion.data.filtering import decimate, downsample


@pytest.mark.parametrize(
    ("size", "block_size"),
    [
        ((20, 20, 20), (5, 5, 5),),
        ((21, 21, 21), (5, 5, 5),),
    ],
)
@pytest.mark.parametrize(
    ("zoom_x", ),
    [(1.0, ), (-1.0, ), (2.0, ), (-2.0, )],
)
@pytest.mark.parametrize(
    ("zoom_y", ),
    [(1.0, ), (-1.0, ), (2.0, ), (-2.0, )],
)
@pytest.mark.parametrize(
    ("zoom_z", ),
    [(1.0, ), (-1.0, ), (2.0, ), (-2.0, )],
)
@pytest.mark.parametrize(
    ("angle_x", ),
    [(0.0, ), (0.2, ), (-0.05, )],
)
@pytest.mark.parametrize(
    ("angle_y", ),
    [(0.0, ), (0.2, ), (-0.05, )],
)
@pytest.mark.parametrize(
    ("angle_z", ),
    [(0.0, ), (0.2, ), (-0.05, )],
)
@pytest.mark.parametrize(
    ("offsets", ),
    [
        (None, ),
        ((0.0, 0.0, 0.0),),
    ],
)
def test_decimation(
    tmp_path,
    size,
    block_size,
    zoom_x,
    zoom_y,
    zoom_z,
    angle_x,
    angle_y,
    angle_z,
    offsets,
    outdir,
):
    """Exercise decimation."""

    # Calculate the number of sub-blocks in each dimension
    num_blocks = [s // b for s, b in zip(size, block_size)]

    # Create the empty array
    voxel_array = np.zeros(size, dtype=np.uint16)

    # Fill the array with increasing values based on sub-block position
    current_block = 0
    for k in range(num_blocks[2]):
        for j in range(num_blocks[1]):
            for i in range(num_blocks[0]):
                voxel_array[
                    i * block_size[0]:(i + 1) * block_size[0],
                    j * block_size[1]:(j + 1) * block_size[1],
                    k * block_size[2]:(k + 1) * block_size[2]
                ] = current_block
                current_block += 1

    fname = tmp_path / "test_img.nii.gz"

    affine = np.eye(4)
    affine[:3, :3] = (
        nb.eulerangles.euler2mat(x=angle_x, y=angle_y, z=angle_z)
        @ np.diag((zoom_x, zoom_y, zoom_z))
        @ affine[:3, :3]
    )

    if offsets is None:
        affine[:3, 3] = -0.5 * nb.affines.apply_affine(affine, np.array(size) - 1)

    test_image = nb.Nifti1Image(voxel_array.astype(np.uint16), affine, None)
    test_image.header.set_data_dtype(np.uint16)
    test_image.to_filename(fname)

    # Need to define test oracle. For now, just see if it doesn't smoke.
    out = decimate(fname, factor=2, smooth=False)

    out = downsample(fname, shape=(10, 10, 10), smooth=False, order=0)

    if outdir:
        from niworkflows.interfaces.reportlets.registration import (
            SimpleBeforeAfterRPT as SimpleBeforeAfter,
        )

        out.to_filename(tmp_path / "decimated.nii.gz")

        SimpleBeforeAfter(
            after_label="Decimated",
            before_label="Original",
            after=str(tmp_path / "decimated.nii.gz"),
            before=str(fname),
            out_report=str(outdir / f'decimated-{tmp_path.name}.svg'),
        ).run()

        out.to_filename(tmp_path / "downsampled.nii.gz")
        SimpleBeforeAfter(
            after_label="Downsampled",
            before_label="Original",
            after=str(tmp_path / "downsampled.nii.gz"),
            before=str(fname),
            out_report=str(outdir / f'downsampled-{tmp_path.name}.svg'),
        ).run()
