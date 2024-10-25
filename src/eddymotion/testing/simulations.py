# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
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
"""Utilities for testing purposes."""

from __future__ import annotations

import nibabel as nib
import numpy as np
from dipy.core.geometry import sphere2cart
from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere, Sphere, disperse_charges
from dipy.sims.voxel import all_tensor_evecs, multi_tensor, single_tensor

# Bounds defined following Canales-Rodriguez, NIMG 184 2019, https://doi.org/10.1016/j.neuroimage.2018.08.071
BOUNDS_LAMBDA1: tuple[float, float] = (1.4e-3, 1.8e-3)
BOUNDS_LAMBDA23: tuple[float, float] = (0.1e-3, 0.5e-3)

BOUNDS_2FIBERS_NONDOMINANT_VF1: tuple[float, float] = (0.3, 0.7)

BOUNDS_2FIBERS_DOMINANT_VF1: tuple[float, float] = (0.1, 0.3)

BOUNDS_3FIBERS_VF1: tuple[float, float] = (0.25, 0.3)
BOUNDS_3FIBERS_VF2: tuple[float, float] = (0.3, 0.35)


def add_b0(bvals: np.ndarray, bvecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert a b=0 at the front of the gradient table (b-values and b-vectors).

    Parameters
    ----------
    bvals : :obj:`~numpy.ndarray`
        Array of b-values.
    bvecs : :obj:`~numpy.ndarray`
        Array of b-vectors.

    Returns
    -------
    :obj:`tuple`
        Updated gradient table (b-values, b-vectors) including a b=0.

    """
    _bvals = np.insert(bvals, 0, 0)
    _bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)
    return _bvals, _bvecs


def create_single_fiber_evecs(theta: float = 0, phi: float = 0) -> np.ndarray:
    """
    Create eigenvectors for a simulated fiber given the polar coordinates of its pricipal axis.

    Parameters
    ----------
    theta : :obj:`float`
        Theta coordinate
    phi : :obj:`float`
        Phi coordinate

    Returns
    -------
    :obj:`~numpy.ndarray`
        Eigenvectors for a single fiber.

    """
    # Polar coordinates (theta, phi) of the principal axis of the tensor
    angles = np.array([theta, phi])
    sticks = np.array(sphere2cart(1, np.deg2rad(angles[0]), np.deg2rad(angles[1])))
    evecs = all_tensor_evecs(sticks)
    return evecs


def create_random_polar_coordinates(
    hsph_dirs: int, seed: int = 1234
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create random polar coordinates.

    Parameters
    ----------
    hsph_dirs : :obj:`int`
        Number of hemisphere directions.
    seed : :obj:`int`, optional
        Seed for the random number generator.

    Returns
    -------
    :obj:`tuple`
        Theta and Phi values of polar coordinates.

    """
    rng = np.random.default_rng(seed)
    theta = np.pi * rng.random(hsph_dirs)
    phi = 2 * np.pi * rng.random(hsph_dirs)
    return theta, phi


def create_diffusion_encoding_gradient_dirs(
    hsph_dirs: int, iterations: int = 5000, seed: int = 1234
) -> Sphere:
    """
    Create the dMRI gradient-encoding directions.

    Parameters
    ----------
    hsph_dirs : :obj:`int`
        Number of hemisphere directions.
    iterations : :obj:`int`, optional
        Number of iterations for charge dispersion.
    seed : :obj:`int`, optional
        Seed for the random number generator.

    Returns
    -------
    :obj:`~dipy.core.sphere.Sphere`
        A sphere with diffusion-encoding gradient directions.

    """
    # Create the gradient-encoding directions placing random points on a hemisphere
    theta, phi = create_random_polar_coordinates(hsph_dirs, seed=seed)
    hsph_initial = HemiSphere(theta=theta, phi=phi)

    # Move the points so that the electrostatic potential energy is minimized
    hsph_updated, _ = disperse_charges(hsph_initial, iterations)

    # Create a sphere
    return Sphere(xyz=np.vstack((hsph_updated.vertices, -hsph_updated.vertices)))


def create_single_shell_gradient_table(
    hsph_dirs: int, bval_shell: float, iterations: int = 5000
) -> gradient_table:
    """
    Create a single-shell gradient table.

    Parameters
    ----------
    hsph_dirs : :obj:`int`
        Number of hemisphere directions.
    bval_shell : :obj:`float`
        Shell b-value.
    iterations : :obj:`int`, optional
        Number of iterations for charge dispersion.

    Returns
    -------
    :obj:`~dipy.core.gradients.gradient_table`
        The gradient table for the single-shell.

    """
    # Create diffusion-encoding gradient directions
    sph = create_diffusion_encoding_gradient_dirs(hsph_dirs, iterations=iterations)

    # Create the gradient bvals and bvecs
    vertices = sph.vertices
    values = np.ones(vertices.shape[0])
    bvecs = vertices
    bvals = bval_shell * values

    # Add a b0 value to the gradient table
    bvals, bvecs = add_b0(bvals, bvecs)
    return gradient_table(bvals, bvecs)


def get_query_vectors(
    gtab: gradient_table, train_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the diffusion-encoding gradient vectors for estimation from the gradient table.

    Get the diffusion-encoding gradient vectors where the signal is to be estimated from the
    gradient table and the training mask: the vectors of interest are those that are masked in
    the training mask. b0 values are excluded.

    Parameters
    ----------
    gtab : :obj:`~dipy.core.gradients.gradient_table`
        Gradient table.
    train_mask : :obj:`~numpy.ndarray`
        Mask for selecting training vectors.

    Returns
    -------
    :obj:`tuple`
        Gradient vectors and indices for estimation.

    """
    idx = np.logical_and(~train_mask, ~gtab.b0s_mask)
    return gtab.bvecs[idx], np.where(idx)[0]


def single_fiber_voxel(gtab, S0, evals, rng, theta=0, phi=0, snr=20):
    # create eigenvectors for a single fiber
    evecs = create_single_fiber_evecs(theta=theta, phi=phi)

    # Generate some data
    return single_tensor(gtab, S0=S0, evals=evals, evecs=evecs, snr=snr, rng=rng)


def create_random_polar_angles(size, rng):
    """Create polar angles drawn from a uniform distribution."""

    return zip(
        rng.uniform(0, np.pi, size=size),
        rng.uniform(0, 2.0 * np.pi, size=size),
        strict=True,
    )


def create_random_diffusivity_eigenvalues(size, rng):
    r"""Create DTI model diffusion tensor eigenvalues ($\lambda_{1},
    \lambda_{2}, \lambda_{3}$) drawn from a uniform distribution."""

    # lambda_2 = lambda_3 following Canales-Rodriguez, NIMG 184 2019,
    # https://doi.org/10.1016/j.neuroimage.2018.08.071
    return zip(
        rng.uniform(*BOUNDS_LAMBDA1, size=size),
        *[rng.uniform(*BOUNDS_LAMBDA23, size=size)] * 2,
        strict=True,
    )


def create_three_fiber_random_volume_fractions(size, rng):
    """Create fiber volume fractions drawn from a uniform distribution for a
    three-fiber configuration."""

    # f1 ~ U(0.25, 0.3) f2 ~ U(0.3, 0.35) and f3 = 1 - (f1 + f2) set
    # according to Canales-Rodriguez, NIMG 184 2019,
    # https://doi.org/10.1016/j.neuroimage.2018.08.071
    f1 = rng.uniform(*BOUNDS_3FIBERS_VF1, size=size)
    f2 = rng.uniform(*BOUNDS_3FIBERS_VF2, size=size)
    return zip(f1 * 100, f2 * 100, (1 - (f1 + f2)) * 100, strict=True)


def create_two_fiber_nondominant_random_volume_fractions(size, rng):
    """Create fiber volume fractions drawn from a uniform distribution for a
    two-fiber configuration with non-dominant fibers."""

    # f1 ~ U(0.3, 0.7), f2 = 1 - f1 following Canales-Rodriguez, NIMG 184 2019,
    # https://doi.org/10.1016/j.neuroimage.2018.08.071
    f1 = rng.uniform(*BOUNDS_2FIBERS_NONDOMINANT_VF1, size=size)
    return zip(f1 * 100, (1 - f1) * 100, strict=True)


def create_two_fiber_dominant_random_volume_fractions(size, rng):
    """Create fiber volume fractions drawn from a uniform distribution for a
    two-fiber configuration with a dominant fiber."""

    # f1 ~ U(0.1, 0.3), f2 = 1 - f1 following to Canales-Rodriguez, NIMG 184
    # 2019, https://doi.org/10.1016/j.neuroimage.2018.08.071
    f1 = rng.uniform(*BOUNDS_2FIBERS_DOMINANT_VF1, size=size)
    return zip(f1 * 100, (1 - f1) * 100, strict=True)


def group_values(values, group_size):
    return np.asarray([values[i : i + group_size] for i in range(0, len(values), group_size)])


def simulate_one_fiber_multivoxel(gtab, S0, snr, n_voxels, rng, evals=None):
    """Create a single-fiber multi-voxel DWI signal."""

    angles = create_random_polar_angles(n_voxels, rng)
    if evals is None:
        _evals = create_random_diffusivity_eigenvalues(n_voxels, rng)
    else:
        _evals = group_values(evals, 3)
        if _evals.shape[0] == 1 and n_voxels != 1:
            _evals = np.repeat(_evals, n_voxels, axis=0)

    signal = np.vstack(
        [
            single_fiber_voxel(gtab, S0, _eignvls, rng, theta=theta, phi=phi, snr=snr)
            for (theta, phi), _eignvls in zip(angles, _evals, strict=True)
        ]
    )

    return signal


def simulate_voxels(S0, hsph_dirs, bval_shell=1000, snr=20, n_voxels=1, evals=None, seed=None):
    # Create a gradient table for a single-shell
    gtab = create_single_shell_gradient_table(hsph_dirs, bval_shell)

    rng = np.random.default_rng(seed)

    signal = simulate_one_fiber_multivoxel(gtab, S0, snr, n_voxels, rng, evals=evals)

    return signal, gtab


def simulate_two_fiber_multivoxel(gtab, S0, snr, n_voxels, rng, dominant):
    """Create two-fiber multi-voxel DWI signal."""

    evals = zip(
        create_random_diffusivity_eigenvalues(n_voxels, rng),
        create_random_diffusivity_eigenvalues(n_voxels, rng),
        strict=False,
    )
    angles = zip(
        create_random_polar_angles(n_voxels, rng),
        create_random_polar_angles(n_voxels, rng),
        strict=False,
    )

    if dominant:
        fractions = create_two_fiber_dominant_random_volume_fractions(n_voxels, rng)
    else:
        fractions = create_two_fiber_nondominant_random_volume_fractions(n_voxels, rng)

    signal = np.vstack(
        [
            multi_tensor(
                gtab, _eignvls, S0=S0, angles=_angles, fractions=_fractions, snr=snr, rng=rng
            )[0]
            for _angles, _eignvls, _fractions in zip(angles, evals, fractions, strict=True)
        ]
    )

    return signal


def simulate_three_fiber_multivoxel(gtab, S0, snr, n_voxels, rng):
    """Create three-fiber multi-voxel DWI signal."""

    evals = zip(
        create_random_diffusivity_eigenvalues(n_voxels, rng),
        create_random_diffusivity_eigenvalues(n_voxels, rng),
        create_random_diffusivity_eigenvalues(n_voxels, rng),
        strict=False,
    )
    angles = zip(
        create_random_polar_angles(n_voxels, rng),
        create_random_polar_angles(n_voxels, rng),
        create_random_polar_angles(n_voxels, rng),
        strict=False,
    )
    fractions = create_three_fiber_random_volume_fractions(n_voxels, rng)

    signal = np.vstack(
        [
            multi_tensor(
                gtab, _eignvls, S0=S0, angles=_angles, fractions=_fractions, snr=snr, rng=rng
            )[0]
            for _angles, _eignvls, _fractions in zip(angles, evals, fractions, strict=True)
        ]
    )

    return signal


def simulate_multifiber_voxels(S0, hsph_dirs, bval_shell=1000, snr=20, n_voxels=1, seed=None):
    """Create a DWI signal with multiple tensors."""

    # Create a gradient table for a single-shell
    gtab = create_single_shell_gradient_table(hsph_dirs, bval_shell)

    rng = np.random.default_rng(seed)

    # Generate the number of fibers on each voxel from a uniform distribution
    n_fibers = rng.integers(1, 4, size=n_voxels)
    unique, counts = np.unique(n_fibers, return_counts=True)

    signal = []
    # Loop over the voxels to create the signals
    for _unique, _counts in zip(unique, counts, strict=False):
        if _unique == 1:
            _signal = simulate_one_fiber_multivoxel(gtab, S0, snr, n_voxels, rng)
        elif _unique == 2:
            # Set a number of voxels where volume fractions will be similar vs.
            # others with a very dominant fiber
            count_nondominant_vf = rng.integers(1, _counts + 1, size=1).item()
            count_dominant_vf = _counts - count_nondominant_vf
            signal_nondominant_vf = simulate_two_fiber_multivoxel(
                gtab, S0, snr, count_nondominant_vf, rng, False
            )
            signal_dominant_vf = simulate_two_fiber_multivoxel(
                gtab, S0, snr, count_dominant_vf, rng, True
            )
            _signal = np.vstack([signal_nondominant_vf, signal_dominant_vf])
        elif _unique == 3:
            _signal = simulate_three_fiber_multivoxel(gtab, S0, snr, _counts, rng)
        else:
            raise NotImplementedError(
                "Diffusion gradient-encoding signal generation not implemented "
                f"for more than 3 fibers: {_unique}"
            )

        signal.append(_signal)

    # Shuffle voxels
    signal = rng.permutation(np.vstack(signal))

    return signal, gtab


def serialize_dwi(dwi_data, dwi_data_fname, affine: np.ndarray | None = None):
    """Serialize DWI data.

    Parameters
    ----------
    dwi_data : :obj:`~numpy.ndarray`
       DWI data.
    dwi_data_fname : :obj:`str`
        Filename of NIfTI file to save the DWI signal.
    affine : :obj:`~numpy.ndarray`, optional
        Affine matrix. If ``None`` an identity affine matrix is used.
    """

    if affine is None:
        affine = np.eye(4)

    dwi_img = nib.Nifti1Image(dwi_data, affine=affine)
    nib.save(dwi_img, dwi_data_fname)


def serialize_gtab(gtab, bval_data_fname, bvec_data_fname):
    """Serialize dMRI gradient-encoding table data into a pair of b-vals and
    b-vecs files.

    Parameters
    ----------
    gtab : :obj:`~dipy.core.gradients.gradient_table`
        Gradient table.
    bval_data_fname : :obj:`str`
        Filename of NIfTI file to save the diffusion-encoding gradient b-vals.
    bvec_data_fname : :obj:`str`
        Filename of NIfTI file to save the diffusion-encoding gradient b-vecs.
    """

    fmt = "%d"
    np.savetxt(bval_data_fname, gtab.bvals, newline=" ", fmt=fmt)
    fmt = "%.3f"
    np.savetxt(bvec_data_fname, gtab.bvecs.T, fmt=fmt)


def serialize_dmri(
    dwi_data,
    gtab,
    dwi_data_fname,
    bval_data_fname,
    bvec_data_fname,
    affine: np.ndarray | None = None,
):
    """Serialize dMRI data.

    Parameters
    ----------
    dwi_data : :obj:`~numpy.ndarray`
       DWI data.
    gtab : :obj:`~dipy.core.gradients.gradient_table`
        Gradient table.
    dwi_data_fname : :obj:`str`
        Filename of NIfTI file to save the DWI signal.
    bval_data_fname : :obj:`str`
        Filename of NIfTI file to save the diffusion-encoding gradient b-vals.
    bvec_data_fname : :obj:`str`
        Filename of NIfTI file to save the diffusion-encoding gradient b-vecs.
    affine : :obj:`~numpy.ndarray`, optional
        Affine matrix. If ``None`` an identity affine matrix is used.
    """

    serialize_dwi(dwi_data, dwi_data_fname, affine=affine)
    serialize_gtab(gtab, bval_data_fname, bvec_data_fname)
