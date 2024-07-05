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
from collections import namedtuple

import numpy as np
import pytest
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs

from eddymotion.model.dipy import (
    PairwiseOrientationKernel,
    compute_exponential_covariance,
    compute_pairwise_angles,
    compute_spherical_covariance,
)

GradientTablePatch = namedtuple("gtab", ["bvals", "bvecs"])


# No need to use normalized vectors: compute_pairwise_angles takes care of it.
# The [-1, 0, 1].T vector serves as a case where e.g. the angle between vector
# [1, 0, 0] and the former is 135 unless the closest polarity flag is set to
# True, in which case it yields 45
@pytest.mark.parametrize(
    ("bvecs1", "bvecs2", "closest_polarity", "expected"),
    [
        (
            np.array(
                [
                    [1, 0, 0, 1, 1, 0, -1],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1, 1, 1],
                ]
            ),
            None,
            True,
            np.array(
                [
                    [0.0, np.pi / 2, np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 2, np.pi / 4],
                    [np.pi / 2, 0.0, np.pi / 2, np.pi / 4, np.pi / 2, np.pi / 4, np.pi / 2],
                    [np.pi / 2, np.pi / 2, 0.0, np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 4],
                    [np.pi / 4, np.pi / 4, np.pi / 2, 0.0, np.pi / 3, np.pi / 3, np.pi / 3],
                    [np.pi / 4, np.pi / 2, np.pi / 4, np.pi / 3, 0.0, np.pi / 3, np.pi / 2],
                    [np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 3, np.pi / 3, 0.0, np.pi / 3],
                    [np.pi / 4, np.pi / 2, np.pi / 4, np.pi / 3, np.pi / 2, np.pi / 3, 0.0],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 0, 0, 1, 1, 0, -1],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1, 1, 1],
                ]
            ),
            None,
            False,
            np.array(
                [
                    [0.0, np.pi / 2, np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                    [np.pi / 2, 0.0, np.pi / 2, np.pi / 4, np.pi / 2, np.pi / 4, np.pi / 2],
                    [np.pi / 2, np.pi / 2, 0.0, np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 4],
                    [np.pi / 4, np.pi / 4, np.pi / 2, 0.0, np.pi / 3, np.pi / 3, 2 * np.pi / 3],
                    [np.pi / 4, np.pi / 2, np.pi / 4, np.pi / 3, 0.0, np.pi / 3, np.pi / 2],
                    [np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 3, np.pi / 3, 0.0, np.pi / 3],
                    [
                        3 * np.pi / 4,
                        np.pi / 2,
                        np.pi / 4,
                        2 * np.pi / 3,
                        np.pi / 2,
                        np.pi / 3,
                        0.0,
                    ],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 0, 0, 1, 1, 0, -1],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1, 1, 1],
                ]
            ),
            np.array(
                [
                    [1, -1],
                    [0, 0],
                    [0, 1],
                ]
            ),
            True,
            np.array(
                [
                    [0.0, np.pi / 4],
                    [np.pi / 2, np.pi / 2],
                    [np.pi / 2, np.pi / 4],
                    [np.pi / 4, np.pi / 3],
                    [np.pi / 4, np.pi / 2],
                    [np.pi / 2, np.pi / 3],
                    [np.pi / 4, 0.0],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 0, 0, 1, 1, 0, -1],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1, 1, 1],
                ]
            ),
            np.array(
                [
                    [1, -1],
                    [0, 0],
                    [0, 1],
                ]
            ),
            False,
            np.array(
                [
                    [0.0, 3 * np.pi / 4],
                    [np.pi / 2, np.pi / 2],
                    [np.pi / 2, np.pi / 4],
                    [np.pi / 4, 2 * np.pi / 3],
                    [np.pi / 4, np.pi / 2],
                    [np.pi / 2, np.pi / 3],
                    [3 * np.pi / 4, 0.0],
                ]
            ),
        ),
    ],
)
def test_compute_pairwise_angles(bvecs1, bvecs2, closest_polarity, expected):
    # DIPY requires the vectors to be normalized
    _bvecs1 = bvecs1 / np.linalg.norm(bvecs1, axis=0)
    gtab1 = gradient_table([1000] * _bvecs1.shape[-1], _bvecs1)

    _bvecs2 = None
    gtab2 = None
    if bvecs2 is not None:
        _bvecs2 = bvecs2 / np.linalg.norm(bvecs2, axis=0)
        gtab2 = gradient_table([1000] * _bvecs2.shape[-1], _bvecs2)

    obtained = compute_pairwise_angles(gtab1, gtab2, closest_polarity)

    if _bvecs2 is not None:
        assert (_bvecs1.shape[-1], _bvecs2.shape[-1]) == obtained.shape
    assert obtained.shape == expected.shape
    np.testing.assert_array_almost_equal(obtained, expected, decimal=2)


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


def test_kernel(repodata):
    """Check kernel construction."""

    bvals, bvecs = read_bvals_bvecs(
        str(repodata / "ds000114_singleshell.bval"),
        str(repodata / "ds000114_singleshell.bvec"),
    )
    gtab_original = gradient_table(bvals, bvecs)

    bvals = gtab_original.bvals[~gtab_original.b0s_mask]
    bvecs = gtab_original.bvecs[~gtab_original.b0s_mask, :]
    gtab = gradient_table(bvals, bvecs)

    kernel = PairwiseOrientationKernel()

    K = kernel(gtab)

    assert K.shape == (len(bvals), len(bvals))
    assert np.allclose(np.diagonal(K), kernel.diag(gtab))

    # DIPY bug - gradient tables cannot be created with just one bvec/bval
    # https://github.com/dipy/dipy/issues/3283
    gtab_Y = GradientTablePatch(bvals[10], bvecs[10, ...])

    K_predict = kernel(gtab, gtab_Y)

    assert K_predict.shape == (K.shape[0],)

    gtab_Y = gradient_table(bvals[10:14], bvecs[10:14, ...])

    K_predict = kernel(gtab, gtab_Y)

    assert K_predict.shape == (K.shape[0], 4)
