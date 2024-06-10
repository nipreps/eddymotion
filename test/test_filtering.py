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

from eddymotion.data.filtering import decimate


@pytest.mark.parametrize(
    ("size", "block_size"),
    [
        ((20, 20, 20), (5, 5, 5),)
    ],
)
def test_decimation(tmp_path, size, block_size):
    """Exercise decimation."""

    # Calculate the number of sub-blocks in each dimension
    num_blocks = [s // b for s, b in zip(size, block_size)]

    # Create the empty array
    voxel_array = np.zeros(size, dtype=int)

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

    nb.Nifti1Image(voxel_array, None, None).to_filename(fname)

    # Need to define test oracle. For now, just see if it doesn't smoke.
    decimate(fname, factor=2, smooth=False, order=1)

    # out.to_filename(tmp_path / "decimated.nii.gz")

    # import pdb; pdb.set_trace()