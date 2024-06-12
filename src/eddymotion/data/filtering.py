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
"""Filtering data."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import ball

DEFAULT_DTYPE = "int16"
"""The default image's data type."""


def advanced_clip(
    data: np.ndarray,
    p_min: float = 35,
    p_max: float = 99.98,
    nonnegative: bool = True,
    dtype: str | np.dtype = DEFAULT_DTYPE,
    invert: bool = False,
) -> np.ndarray:
    """
    Clips outliers from a n-dimensional array and scales/casts to a specified data type.

    This function removes outliers from both ends of the intensity distribution
    in a n-dimensional array using percentiles. It optionally enforces non-negative
    values and scales the data to fit within a specified data type (e.g., uint8
    for image registration). To remove outliers more robustly, the function
    first applies a median filter to the data before calculating clipping thresholds.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        The input n-dimensional data array.
    p_min : :obj:`float`, optional
        The lower percentile threshold for clipping. Values below this percentile
        are set to the threshold value.
    p_max : :obj:`float`, optional
        The upper percentile threshold for clipping. Values above this percentile
        are set to the threshold value.
    nonnegative : :obj:`bool`, optional
        If True, only consider non-negative values when calculating thresholds.
    dtype : :obj:`str` or :obj:`~numpy.dtype`, optional
        The desired data type for the output array. Supported types are "uint8"
        and "int16".
    invert : :obj:`bool`, optional
        If ``True``, inverts the intensity values after scaling (1.0 - ``data``).

    Returns
    -------
    :obj:`~numpy.ndarray`
        The clipped and scaled data array with the specified data type.

    """

    # Calculate stats on denoised version to avoid outlier bias
    denoised = median_filter(data, footprint=ball(3))

    a_min = np.percentile(denoised[denoised >= 0] if nonnegative else denoised, p_min)
    a_max = np.percentile(denoised[denoised >= 0] if nonnegative else denoised, p_max)

    # Clip and scale data
    data = np.clip(data, a_min=a_min, a_max=a_max)
    data -= data.min()
    data /= data.max()

    if invert:
        data = 1.0 - data

    if dtype in ("uint8", "int16"):
        data = np.round(255 * data).astype(dtype)

    return data
