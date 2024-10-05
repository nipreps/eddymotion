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
import numpy as np
import pytest

from eddymotion.math.utils import compute_angle, is_positive_definite


def test_is_positive_definite():
    matrix = np.array([[4, 1, 2], [1, 3, 1], [2, 1, 5]])
    assert is_positive_definite(matrix)

    matrix = np.array([[4, 1, 2], [1, -3, 1], [2, 1, 5]])
    assert not is_positive_definite(matrix)


@pytest.mark.parametrize(
    ("v1", "v2", "closest_polarity", "expected"),
    [
        ([1, 0, 0], [1, 0, 0], False, 0),
        ([1, 0, 0], [0, 1, 0], False, np.pi / 2),
        ([1, 0, 0], [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)], False, np.pi * 3 / 4),
        ([1, 0, 0], [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)], True, np.pi / 4),
    ],
)
def test_compute_angle(v1, v2, closest_polarity, expected):
    obtained = compute_angle(v1, v2, closest_polarity=closest_polarity)
    assert np.isclose(obtained, expected)
