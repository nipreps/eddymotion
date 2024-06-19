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
import numpy as np


def compute_pairwise_angles(bvecs1, bvecs2, closest_polarity):
    r"""Compute pairwise angles across diffusion gradient encoding directions.

    Following [Andersson15]_, it computes the smallest of the angles between
    each pair if ``closest_polarity`` is ``True``, i.e.

    .. math::

        \theta(\mathbf{g}, \mathbf{g'}) = \arccos(\abs{\langle \mathbf{g}, \mathbf{g'} \rangle})

    Parameters
    ----------
    bvecs1 : :obj:`~numpy.ndarray`
        Diffusion gradient encoding directions in FSL format.
    bvecs2 : :obj:`~numpy.ndarray`
        Diffusion gradient encoding directions in FSL format.
    closest_polarity : :obj:`bool`
        ``True`` to consider the smallest of the two angles between the crossing
         lines resulting from reversing each vector pair.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Pairwise angles across diffusion gradient encoding directions.

    Examples
    --------
    >>> compute_pairwise_angles(
    ...     ((1.0, -1.0), (0.0, 0.0), (0.0, 0.0)),
    ...     ((1.0, -1.0), (0.0, 0.0), (0.0, 0.0)),
    ...     False,
    ... )[0, 1]  # doctest: +ELLIPSIS
    3.1415...
    >>> compute_pairwise_angles(
    ...     ((1.0, -1.0), (0.0, 0.0), (0.0, 0.0)),
    ...     ((1.0, -1.0), (0.0, 0.0), (0.0, 0.0)),
    ...     True,
    ... )[0, 1]
    0.0

    References
    ----------
    .. [Andersson15] J. L. R. Andersson. et al., An integrated approach to
       correction for off-resonance effects and subject movement in diffusion MR
       imaging, NeuroImage 125 (2016) 1063–1078
    """

    if np.shape(bvecs1)[0] != 3:
        raise ValueError(f"bvecs1 must be of shape (3, N). Found: {bvecs1.shape}")

    if np.shape(bvecs2)[0] != 3:
        raise ValueError(f"bvecs2 must be of shape (3, N). Found: {bvecs2.shape}")

    # Ensure b-vectors are unit-norm
    bvecs1 = np.array(bvecs1) / np.linalg.norm(bvecs1, axis=0)
    bvecs2 = np.array(bvecs2) / np.linalg.norm(bvecs2, axis=0)
    cosines = np.clip(bvecs1.T @ bvecs2, -1.0, 1.0)
    return np.arccos(np.abs(cosines) if closest_polarity else cosines)
