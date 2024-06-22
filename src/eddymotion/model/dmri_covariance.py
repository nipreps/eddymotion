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


def _ensure_positive_scale(a):
    if a <= 0:
        raise ValueError(f"a must be strictly positive. Provided: {a}")


def compute_exponential_covariance(theta, a):
    r"""Compute the exponential covariance matrix following eq. 9 in [Andersson15]_.

    .. math::

        C(\theta) = \exp(- \frac{\theta}{a})

    Parameters
    ----------
    theta : :obj:`~numpy.ndarray`
        Pairwise angles across diffusion gradient encoding directions.
    a : :obj:`float`
        Positive scale parameter that here determines the "distance" at which θ
        the covariance goes to zero.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Exponential covariance function values.
    """

    _ensure_positive_scale(a)

    return np.exp(-theta / a)


def compute_spherical_covariance(theta, a):
    r"""Compute the spherical covariance matrix following eq. 10 in [Andersson15]_.

    .. math::

        C(\theta) = \begin{cases}
            1 - \frac{3 \theta}{2 a} + \frac{\theta^3}{2 a^3} & \textnormal{if} \; \theta \leq a \\
            0 & \textnormal{if} \; \theta > a
        \end{cases}

    Parameters
    ----------
    theta : :obj:`~numpy.ndarray`
        Pairwise angles across diffusion gradient encoding directions.
    a : :obj:`float`
        Positive scale parameter that here determines the "distance" at which θ
        the covariance goes to zero.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Spherical covariance matrix.
    """

    _ensure_positive_scale(a)

    return np.where(theta <= a, 1 - 3 * (theta / a) ** 2 + 2 * (theta / a) ** 3, 0)
