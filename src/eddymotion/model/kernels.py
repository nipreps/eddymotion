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
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel


class SphericalCovarianceKernel(Kernel):
    """
    Custom kernel based on spherical covariance function.

    Parameters
    ----------
    lambda_s : :obj:`float`
        Scale parameter for the covariance function.
    a : :obj:`float`
        Distance parameter where the covariance function goes to zero.
    sigma_sq : :obj:`float`
        Noise variance term.
    lambda_s_bounds : :obj:`tuple` of :obj:`float` or "fixed"
        The lower and upper bound on lambda_s.
    a_bounds : :obj:`tuple` of :obj:`float` or "fixed"
        The lower and upper bound on a.
    sigma_sq_bounds : :obj:`tuple` of :obj:`float` or "fixed"
        The lower and upper bound on sigma_sq.
    """

    def __init__(
        self,
        lambda_s=2.0,
        a=0.1,
        sigma_sq=1.0,
        lambda_s_bounds=(1e-5, 1e4),
        a_bounds=(1e-5, np.pi),
        sigma_sq_bounds=(1e-5, 1e4),
    ):
        self.lambda_s = lambda_s
        self.a = a
        self.sigma_sq = sigma_sq
        self.lambda_s_bounds = lambda_s_bounds
        self.a_bounds = a_bounds
        self.sigma_sq_bounds = sigma_sq_bounds

    @property
    def hyperparameter_lambda_s(self):
        return Hyperparameter("lambda_s", "numeric", self.lambda_s_bounds)

    @property
    def hyperparameter_a(self):
        return Hyperparameter("a", "numeric", self.a_bounds)

    @property
    def hyperparameter_sigma_sq(self):
        return Hyperparameter("sigma_sq", "numeric", self.sigma_sq_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """
        Compute the kernel matrix.

        Parameters
        ----------
        X : :obj:`array-like` of shape (n_samples, n_samples)
            Precomputed pairwise angles.
        Y : :obj:`array-like` of shape (n_samples, n_samples), optional
            Second input for the kernel function.
        eval_gradient : :obj:`bool`
            Evaluate the gradient with respect to the log of
            the kernel hyperparameter.

        Returns
        -------
        K : :obj:`array-like` of shape (n_samples, n_samples)
            Kernel matrix.
        K_gradient : :obj:`array-like` of shape (n_samples, n_samples, n_dims), optional
            The gradient of the kernel matrix with respect to the log of the
            hyperparameters. Only returned when `eval_gradient` is True.
        """
        if Y is not None:
            X = Y
            K = self.lambda_s * np.where(X <= self.a, 1 - 3 * (X / self.a) ** 2 + 2 * (X / self.a) ** 3, 0)
        else:
            K = self.lambda_s * np.where(X <= self.a, 1 - 3 * (X / self.a) ** 2 + 2 * (X / self.a) ** 3, 0) + self.sigma_sq * np.eye(len(X))

        K_gradient = None
        if eval_gradient:
            K_gradient = np.zeros((X.shape[0], X.shape[1], 3))
            dists_deriv = np.zeros_like(X)
            mask = X <= self.a
            dists_deriv[mask] = (3 * X[mask] / self.a**2) - (
                1.5 * (X[mask] / self.a) ** 2
            ) / self.a
            K_gradient[:, :, 0] = K / self.lambda_s
            K_gradient[:, :, 1] = self.lambda_s * dists_deriv
            if Y is not None:
                K_gradient[:, :, 2] = np.zeros(np.shape(X))
            else:
                K_gradient[:, :, 2] = np.eye(len(X))

            return self.lambda_s * K, K_gradient

        return K

    def diag(self, X):
        """
        Returns the diagonal of the kernel matrix.

        Parameters
        ----------
        X : :obj:`array-like` of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :obj:`array-like` of shape (n_samples,)
            Diagonal of the kernel matrix.
        """
        return np.full(X.shape[0], self.lambda_s + self.sigma_sq)

    def is_stationary(self):
        """
        Returns whether the kernel is stationary.

        Returns
        -------
        :obj:`bool`
            True if the kernel is stationary.
        """
        return True

    def get_params(self, deep=True):
        """
        Get parameters of the kernel.

        Parameters
        ----------
        deep : :obj:`bool`
            Whether to return the parameters of the contained subobjects.

        Returns
        -------
        params : :obj:`dict`
            Parameter names mapped to their values.
        """
        return {"lambda_s": self.lambda_s, "a": self.a, "sigma_sq": self.sigma_sq}

    def set_params(self, **params):
        """
        Set parameters of the kernel.

        Parameters
        ----------
        params : :obj:`dict`
            Kernel parameters.

        Returns
        -------
        self : :obj:`object`
            Returns self.
        """
        self.lambda_s = params.get("lambda_s", self.lambda_s)
        self.a = params.get("a", self.a)
        self.sigma_sq = params.get("sigma_sq", self.sigma_sq)
        return self
