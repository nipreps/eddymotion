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

from numbers import Number

import numpy as np
from nibabel import Nifti1Image, load
from nibabel.affines import apply_affine, voxel_sizes
from scipy.ndimage import gaussian_filter as _gs
from scipy.ndimage import map_coordinates, median_filter
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


def gaussian_filter(
    data: np.ndarray,
    vox_width: float | tuple[float, float, float],
) -> np.ndarray:
    """
    Applies a Gaussian smoothing filter to a n-dimensional array.

    This function smooths the input data using a Gaussian filter with a specified
    width (sigma) in voxels along each relevant dimension. It automatically
    handles different data dimensionalities (2D, 3D, or 4D) and ensures that
    smoothing is not applied along the time or orientation dimension (if present
    in 4D data).

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        The input data array.
    vox_width : :obj:`float` or :obj:`tuple` of three :obj:`float`
        The smoothing kernel width (sigma) in voxels. If a single :obj:`float` is provided,
        it is applied uniformly across all spatial dimensions. Alternatively, a
        tuple of three floats can be provided to specify different sigma values
        for each spatial dimension (x, y, z).

    Returns
    -------
    :obj:`~numpy.ndarray`
        The smoothed data array.

    """

    data = np.squeeze(data)  # Drop unused dimensions
    ndim = data.ndim

    if isinstance(vox_width, Number):
        vox_width = tuple([vox_width] * min(3, ndim))

    # Do not smooth across time/orientation (if present in 4D data)
    if ndim == 4 and len(vox_width) == 3:
        vox_width = (*vox_width, 0)

    return _gs(data, vox_width)


def downsample(
    in_file: str,
    shape: tuple[int, int, int],
    smooth: bool | tuple[int, int, int] = True,
    order: int = 3,
    nonnegative: bool = True,
) -> Nifti1Image:
    """
    Downsamples a 3D or 4D Nifti image by a specified downsampling factor.

    This function downsamples a Nifti image by averaging voxels within a user-defined
    factor in each spatial dimension. It optionally applies Gaussian smoothing
    before downsampling to reduce aliasing artifacts. The function also handles
    updating the affine transformation matrix to reflect the change in voxel size.

    Parameters
    ----------
    in_file : :obj:`str`
        Path to the input NIfTI image file.
    factor : :obj:`int` or :obj:`tuple`
        The downsampling factor. If a single integer is provided, it is applied
        uniformly across all spatial dimensions. Alternatively, a tuple of three
        integers can be provided to specify different downsampling factors for each
        spatial dimension (x, y, z). Values must be greater than 0.
    smooth : :obj:`bool` or :obj:`tuple`, optional (default=``True``)
        Controls application of Gaussian smoothing before downsampling. If True,
        a smoothing kernel size equal to the downsampling factor is applied.
        Alternatively, a tuple of three integers can be provided to specify
        different smoothing kernel sizes for each spatial dimension. Setting to
        False disables smoothing.
    order : :obj:`int`, optional (default=3)
        The order of the spline interpolation used for downsampling. Higher
        orders provide smoother results but are computationally more expensive.
    nonnegative : :obj:`bool`, optional (default=``True``)
        If True, negative values in the downsampled data are set to zero.

    Returns
    -------
    :obj:`~nibabel.Nifti1Image`
        The downsampled NIfTI image object.

    """

    imnii = load(in_file)
    data = np.squeeze(imnii.get_fdata())  # Remove unused dimensions
    datashape = np.array(data.shape)
    shape = np.array(shape)
    ndim = data.ndim

    if smooth:
        if smooth is True:
            smooth = datashape[:3] / shape[:3]
        data = gaussian_filter(data, smooth)

    extents = np.abs(
        apply_affine(imnii.affine, datashape - 1)
        - apply_affine(imnii.affine, (0.0, 0.0, 0.0))
    )
    newzooms = extents / shape

    # Update affine transformation
    newaffine = np.eye(4)
    oldzooms = voxel_sizes(imnii.affine)
    newaffine[:3, :3] = np.diag(newzooms / oldzooms) @ imnii.affine[:3, :3]

    # Update offset so new array is aligned with original
    newaffine[:3, 3] = (
        apply_affine(imnii.affine, 0.5 * datashape)
        - apply_affine(newaffine, 0.5 * shape)
    )

    xfm = np.linalg.inv(imnii.affine) @ newaffine

    # Create downsampled grid
    down_grid = np.array(
        np.meshgrid(
            *[np.arange(_s, step=1) for _s in shape],
            indexing="ij",
        )
    )

    # Locations is an Nx3 array of index coordinates of the original image where we sample
    locations = apply_affine(xfm, down_grid.reshape((ndim, np.prod(shape))).T)

    # Resample data on the new grid
    resampled = map_coordinates(
        data,
        locations.T,
        order=order,
        mode="mirror",
        prefilter=True,
    ).reshape(shape)

    # Set negative values to zero (optional)
    if order > 2 and nonnegative:
        resampled[resampled < 0] = 0

    # Create new Nifti image with updated information
    newnii = Nifti1Image(resampled, newaffine, imnii.header)
    newnii.set_sform(newaffine, code=1)
    newnii.set_qform(newaffine, code=1)

    return newnii


def decimate(
    in_file: str,
    factor: int | tuple[int, int, int],
    smooth: bool | tuple[int, int, int] = True,
    nonnegative: bool = True,
) -> Nifti1Image:
    """
    Decimates a 3D or 4D Nifti image by a specified downsampling factor.

    This function downsamples a Nifti image by averaging voxels within a user-defined
    factor in each spatial dimension. It optionally applies Gaussian smoothing
    before downsampling to reduce aliasing artifacts. The function also handles
    updating the affine transformation matrix to reflect the change in voxel size.

    Parameters
    ----------
    in_file : :obj:`str`
        Path to the input NIfTI image file.
    factor : :obj:`int` or :obj:`tuple`
        The downsampling factor. If a single integer is provided, it is applied
        uniformly across all spatial dimensions. Alternatively, a tuple of three
        integers can be provided to specify different downsampling factors for each
        spatial dimension (x, y, z). Values must be greater than 0.
    smooth : :obj:`bool` or :obj:`tuple`, optional (default=``True``)
        Controls application of Gaussian smoothing before downsampling. If True,
        a smoothing kernel size equal to the downsampling factor is applied.
        Alternatively, a tuple of three integers can be provided to specify
        different smoothing kernel sizes for each spatial dimension. Setting to
        False disables smoothing.
    nonnegative : :obj:`bool`, optional (default=``True``)
        If True, negative values in the downsampled data are set to zero.

    Returns
    -------
    :obj:`~nibabel.Nifti1Image`
        The downsampled NIfTI image object.

    """

    imnii = load(in_file)
    data = np.squeeze(imnii.get_fdata())  # Remove unused dimensions
    ndim = data.ndim

    if isinstance(factor, Number):
        factor = tuple([factor] * min(3, ndim))

    if any(f <= 0 for f in factor[:3]):
        raise ValueError("All spatial downsampling factors must be positive.")

    if ndim == 4 and len(factor) == 3:
        factor = (*factor, 0)

    if smooth:
        if smooth is True:
            smooth = factor
        data = gaussian_filter(data, smooth)

    # Update affine transformation
    newaffine = imnii.affine.copy()
    newaffine[:3, :3] = np.array(factor[:3]) * newaffine[:3, :3]

    # Create new Nifti image with updated information
    newnii = Nifti1Image(
        data[::factor[0], ::factor[1], ::factor[2]],
        newaffine,
        imnii.header,
    )
    newnii.set_sform(newaffine, code=1)
    newnii.set_qform(newaffine, code=1)

    return newnii
