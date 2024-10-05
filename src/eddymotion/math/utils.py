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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY kIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
import numpy as np


def is_positive_definite(matrix):
    """Check whether the given matrix is positive definite. Any positive
    definite matrix can be decomposed as the product of a lower triangular
    matrix and its conjugate transpose by performing the Cholesky decomposition.
    Parameters
    ----------
    matrix : :obj:`~numpy.ndarray`
        The matrix to check.
    Returns
    -------
    True is the matrix is positive definite; False otherwise
    """

    try:
        # Attempt Cholesky decomposition
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        # Matrix is not positive definite
        return False


def compute_angle(v1, v2, closest_polarity=False):
    """Compute the angle between two vectors.

    Parameters
    ----------
    v1 : :obj:`~numpy.ndarray`
        First vector.
    v2 : :obj:`~numpy.ndarray`
        Second vector.
    closest_polarity : :obj:`bool`
        ``True`` to consider the smallest of the two angles between the crossing
         lines resulting from reversing both vectors.

    Returns
    -------
    :obj:`float`
        The angle between the two vectors in radians.

    Examples
    --------
    >>> compute_angle(
    ...     np.array((1.0, 0.0, 0.0)),
    ...     np.array((-1.0, 0.0, 0.0)),
    ... )  # doctest: +ELLIPSIS
    3.1415...
    >>> compute_angle(
    ...     np.array((1.0, 0.0, 0.0)),
    ...     np.array((-1.0, 0.0, 0.0)),
    ...     closest_polarity=True,
    ... )
    0.0

    """

    cosine_angle = (v1 / np.linalg.norm(v1)) @ (v2 / np.linalg.norm(v2))
    # Clip values to handle numerical errors
    cosine_angle = np.clip(
        np.abs(cosine_angle) if closest_polarity else cosine_angle,
        -1.0,
        1.0,
    )
    return np.arccos(cosine_angle)
