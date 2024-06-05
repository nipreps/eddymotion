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

from eddymotion.model.kernels import SphericalCovarianceKernel


def test_kernel_call():
    # Create a SphericalCovarianceKernel instance
    kernel = SphericalCovarianceKernel(lambda_=2.0, a=1.0, sigma_sq=0.5)

    # Create trivial data (pairwise angles)
    theta = np.array([[0.0, 0.5, 1.0],
                      [0.5, 0.0, 0.5],
                      [1.0, 0.5, 0.0]])

    # Expected kernel matrix
    expected_K = np.array([[2.5, 2.0 * (1 - 3 * (0.5 / 1.0) ** 2 + 2 * (0.5 / 1.0) ** 3), 0.0],
                           [2.0 * (1 - 3 * (0.5 / 1.0) ** 2 + 2 * (0.5 / 1.0) ** 3), 2.5, 2.0 * (1 - 3 * (0.5 / 1.0) ** 2 + 2 * (0.5 / 1.0) ** 3)],
                           [0.0, 2.0 * (1 - 3 * (0.5 / 1.0) ** 2 + 2 * (0.5 / 1.0) ** 3), 2.5]])

    # Compute the kernel matrix using the kernel instance
    K = kernel(theta)

    # Assert the kernel matrix is as expected
    np.testing.assert_array_almost_equal(K, expected_K, decimal=6)

    # Check if the kernel matrix is positive definite
    eigenvalues = np.linalg.eigvals(K)
    assert np.all(eigenvalues > 0), "Kernel matrix is not positive definite"


def test_kernel_diag():
    # Create a SphericalCovarianceKernel instance
    kernel = SphericalCovarianceKernel(lambda_=2.0, a=1.0, sigma_sq=0.5)

    # Create trivial data
    X = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

    # Expected diagonal
    expected_diag = np.array([2.5, 2.5, 2.5])

    # Compute the diagonal using the kernel instance
    diag = kernel.diag(X)

    # Assert the diagonal is as expected
    np.testing.assert_array_almost_equal(diag, expected_diag, decimal=6)


if __name__ == "__main__":
    pytest.main()
