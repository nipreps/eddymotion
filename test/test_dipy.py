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
from collections import namedtuple

import numpy as np
import pytest
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs

from eddymotion.model._dipy import (
    PairwiseOrientationKernel,
    compute_exponential_covariance,
    compute_pairwise_angles,
    compute_spherical_covariance,
)

GradientTablePatch = namedtuple("gtab", ["bvals", "bvecs"])


THETAS = np.linspace(0, np.pi / 2, num=50)
EXPECTED_EXPONENTIAL = [
    1.0,
    0.93789795,
    0.87965256,
    0.82502433,
    0.77378862,
    0.72573476,
    0.68066514,
    0.63839443,
    0.59874883,
    0.5615653,
    0.52669094,
    0.49398235,
    0.46330503,
    0.43453284,
    0.40754745,
    0.38223792,
    0.35850016,
    0.33623656,
    0.31535558,
    0.29577135,
    0.27740334,
    0.26017603,
    0.24401856,
    0.22886451,
    0.21465155,
    0.20132125,
    0.18881879,
    0.17709275,
    0.16609493,
    0.15578009,
    0.14610583,
    0.13703236,
    0.12852236,
    0.12054086,
    0.11305503,
    0.10603408,
    0.09944914,
    0.09327315,
    0.08748069,
    0.08204796,
    0.07695262,
    0.0721737,
    0.06769156,
    0.06348778,
    0.05954506,
    0.05584719,
    0.05237896,
    0.04912612,
    0.04607529,
    0.04321392,
]
EXPECTED_SPHERICAL = [
    1.00000000e00,
    9.60914866e-01,
    9.21882843e-01,
    8.82957040e-01,
    8.44190567e-01,
    8.05636535e-01,
    7.67348053e-01,
    7.29378232e-01,
    6.91780182e-01,
    6.54607013e-01,
    6.17911835e-01,
    5.81747758e-01,
    5.46167893e-01,
    5.11225349e-01,
    4.76973237e-01,
    4.43464666e-01,
    4.10752747e-01,
    3.78890590e-01,
    3.47931306e-01,
    3.17928003e-01,
    2.88933793e-01,
    2.61001786e-01,
    2.34185091e-01,
    2.08536818e-01,
    1.84110079e-01,
    1.60957982e-01,
    1.39133639e-01,
    1.18690159e-01,
    9.96806519e-02,
    8.21582285e-02,
    6.61759987e-02,
    5.17870726e-02,
    3.90445603e-02,
    2.80015720e-02,
    1.87112177e-02,
    1.12266077e-02,
    5.60085192e-03,
    1.88706063e-03,
    1.38343910e-04,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
]


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


def test_compute_exponential_covariance():
    obtained = compute_exponential_covariance(THETAS, 0.5)
    assert np.allclose(obtained, EXPECTED_EXPONENTIAL)


def test_compute_spherical_covariance():
    obtained = compute_spherical_covariance(THETAS, 1.23)
    assert np.allclose(obtained, EXPECTED_SPHERICAL)


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

    assert K_predict.shape == (K.shape[0], 1)

    gtab_Y = gradient_table(bvals[10:14], bvecs[10:14, ...])

    K_predict = kernel(gtab, gtab_Y)

    assert K_predict.shape == (K.shape[0], 4)
