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

from eddymotion.model.gradient_utils import compute_pairwise_angles


def test_compute_pairwise_angles():
    # No need to use normalized vectors: compute_angle takes care of dealing
    # with it.
    # The last vector serves as a case where e.g. the angle between the first
    # vector and the last one is 135, and the method yielding the smallest
    # resulting angle between the crossing lines (45 vs 135)
    bvecs = np.array(
        [
            [1, 0, 0, 1, 1, 0, -1],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 1, 1],
        ]
    )

    expected = np.array(
        [
            [0.0, np.pi / 2, np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 2, np.pi / 4],
            [np.pi / 2, 0.0, np.pi / 2, np.pi / 4, np.pi / 2, np.pi / 4, np.pi / 2],
            [np.pi / 2, np.pi / 2, 0.0, np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 4],
            [np.pi / 4, np.pi / 4, np.pi / 2, 0.0, np.pi / 3, np.pi / 3, np.pi / 3],
            [np.pi / 4, np.pi / 2, np.pi / 4, np.pi / 3, 0.0, np.pi / 3, np.pi / 2],
            [np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 3, np.pi / 3, 0.0, np.pi / 3],
            [np.pi / 4, np.pi / 2, np.pi / 4, np.pi / 3, np.pi / 2, np.pi / 3, 0.0],
        ]
    )

    smallest = True
    obtained = compute_pairwise_angles(bvecs, smallest)

    # Expect N*N elements
    assert bvecs.shape[-1] ** 2 == np.prod(obtained.shape)
    assert obtained.shape == expected.shape
    # Check that the matrix is symmetric
    assert np.allclose(expected, expected.T)
    np.testing.assert_array_almost_equal(obtained, expected, decimal=2)
