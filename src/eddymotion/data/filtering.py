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


def gaussian_filter(data, vox_width):
    """
    Apply a Gaussian smoothing filter of a given width (in voxels)

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        The input image's data array
    vox_width : :obj:`numbers.Number` or :obj:`tuple` or :obj:`list`
        The smoothing kernel width in voxels

    Returns
    -------
    data : :obj:`numpy.ndarray`
        The smoothed dataset

    """
    from numbers import Number
    import numpy as np
    from scipy.ndimage import gaussian_filter as _gs

    data = np.squeeze(data)  # drop unused dimensions
    ndim = data.ndim

    if isinstance(vox_width, Number):
        vox_width = tuple([vox_width] * min(3, ndim))

    # Do not smooth across time/orientation
    if ndim == 4 and len(vox_width) == 3:
        vox_width = (*vox_width, 0)

    return _gs(data, vox_width)


def decimate(in_file, factor, smooth=True, order=3, nonnegative=True):
    from numbers import Number
    import numpy as np
    from scipy.ndimage import map_coordinates
    import nibabel as nb

    imnii = nb.load(in_file)
    data = np.squeeze(imnii.get_fdata())
    datashape = data.shape
    ndim = data.ndim

    if isinstance(factor, Number):
        factor = tuple([factor] * min(3, ndim))

    if ndim == 4 and len(factor) == 3:
        factor = (*factor, 0)

    if smooth:
        if smooth is True:
            smooth = factor

        data = gaussian_filter(data, smooth)

    down_grid = np.array(
        np.meshgrid(
            *[np.arange(_s, step=int(_f) or 1) for _s, _f in zip(datashape, factor)],
            indexing="ij",
        )
    )
    new_shape = down_grid.shape[1:]
    newaffine = imnii.affine.copy()
    newaffine[:3, :3] = np.array(factor[:3]) * newaffine[:3, :3]
    # newaffine[:3, 3] += imnii.affine[:3, :3] @ (0.5 / np.array(factor[:3], dtype="float32"))

    # Resample data in the new grid
    resampled = map_coordinates(
        data,
        down_grid.reshape((ndim, np.prod(new_shape))),
        order=order,
        mode="constant",
        cval=0,
        prefilter=True,
    ).reshape(new_shape)

    if order > 2 and nonnegative:
        resampled[resampled < 0] = 0

    newnii = nb.Nifti1Image(resampled, newaffine, imnii.header)
    newnii.set_sform(newaffine, code=1)
    newnii.set_qform(newaffine, code=1)
    return newnii


def advanced_clip(
    data, p_min=35, p_max=99.98, nonnegative=True, dtype="int16", invert=False
):
    """
    Remove outliers at both ends of the intensity distribution and fit into a given dtype.

    This interface tries to emulate ANTs workflows' massaging that truncate images into
    the 0-255 range, and applies percentiles for clipping images.
    For image registration, normalizing the intensity into a compact range (e.g., uint8)
    is generally advised.

    To more robustly determine the clipping thresholds, spikes are removed from data with
    a median filter.
    Once the thresholds are calculated, the denoised data are thrown away and the thresholds
    are applied on the original image.

    """
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import ball

    # Calculate stats on denoised version, to preempt outliers from biasing
    denoised = ndimage.median_filter(data, footprint=ball(3))

    a_min = np.percentile(denoised[denoised >= 0] if nonnegative else denoised, p_min)
    a_max = np.percentile(denoised[denoised >= 0] if nonnegative else denoised, p_max)

    # Clip and cast
    data = np.clip(data, a_min=a_min, a_max=a_max)
    data -= data.min()
    data /= data.max()

    if invert:
        data = 1.0 - data

    if dtype in ("uint8", "int16"):
        data = np.round(255 * data).astype(dtype)

    return data
