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
import pytest

from eddymotion.model.dmri_covariance import (
    compute_exponential_covariance,
    compute_spherical_covariance,
)


@pytest.mark.parametrize(
    ("theta", "a", "expected"),
    [
        (
            np.asarray(
                [0.0, np.pi / 2, np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 2, np.pi / 4],
            ),
            1.0,
            np.asarray(
                [1.0, 0.20787958, 0.20787958, 0.45593813, 0.45593813, 0.20787958, 0.45593813]
            ),
        )
    ],
)
def test_compute_exponential_covariance(theta, a, expected):
    obtained = compute_exponential_covariance(theta, a)
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize(
    ("theta", "a", "expected"),
    [
        (
            np.asarray(
                [0.0, np.pi / 2, np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 2, np.pi / 4],
            ),
            1.0,
            np.asarray([1.0, 0.0, 0.0, 0.11839532, 0.11839532, 0.0, 0.11839532]),
        )
    ],
)
def test_compute_spherical_covariance(theta, a, expected):
    obtained = compute_spherical_covariance(theta, a)
    assert np.allclose(obtained, expected)
