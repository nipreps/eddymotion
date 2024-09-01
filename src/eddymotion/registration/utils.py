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
"""Utilities to aid in performing and evaluating image registration."""

from __future__ import annotations

from itertools import product

import nibabel as nb
import nitransforms as nt
import numpy as np


def displacements_within_mask(
    maskimg: nb.spatialimages.SpatialImage,
    test_xfm: nt.base.BaseTransform,
    reference_xfm: nt.base.BaseTransform | None = None,
) -> np.ndarray:
    maskdata = np.asanyarray(maskimg.dataobj) > 0
    xyz = nb.affines.apply_affine(
        maskimg.affine,
        np.argwhere(maskdata),
    )
    targets = test_xfm.map(xyz)

    diffs = targets - xyz if reference_xfm is None else targets - reference_xfm.map(xyz)
    return np.linalg.norm(diffs, axis=-1)


def displacement_framewise(
    img: nb.spatialimages.SpatialImage,
    test_xfm: nt.base.BaseTransform,
    radius: float = 50.0,
):
    affine = img.affine
    center_ijk = 0.5 * (np.array(img.shape[:3]) - 1)
    center_xyz = nb.affines.apply_affine(affine, center_ijk)
    fd_coords = np.array(list(product(*((radius, -radius),) * 3))) + center_xyz
    return np.mean(np.linalg.norm(test_xfm.map(fd_coords) - fd_coords, axis=-1))
