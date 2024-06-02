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
from sklearn.gaussian_process.kernels import Kernel


class SquaredExponentialCovarianceKernel(Kernel):
    r"""Kernel based on a squared exponential function for Gaussian processes on
    multi-shell DWI data following to eqs. 14 and 16 in [Andersson15]_.

    For a 2-shell case, the $$\mathbf{K}$$ kernel can be written as:

    .. math::

        \begin{equation}
        \mathbf{K} = \left[
            \begin{matrix}
                \lambda C_{\theta}(\theta (\mathbf{G}_{1}); a) + \sigma_{1}^{2} \mathbf{I} & \lambda C_{\theta}(\theta (\mathbf{G}_{2}, \mathbf{G}_{1}); a) C_{b}(b_{2}, b_{1}; \ell) \\
                \lambda C_{\theta}(\theta (\mathbf{G}_{1}, \mathbf{G}_{2}); a) C_{b}(b_{1}, b_{2}; \ell) & \lambda C_{\theta}(\theta (\mathbf{G}_{2}); a) + \sigma_{2}^{2} \mathbf{I} \\
            \end{matrix}
        \right]
        \end{equation}

    Parameters
    ----------
    lambda_ : float
        Scale parameter for the covariance function.
    a : float
        Distance parameter where the covariance function goes to zero.
    sigma_sq : float
        Noise variance term.
    """

    def __init__(self, lambda_=1.0, a=1.0, sigma_sq=1.0):
        self.lambda_ = lambda_
        self.a = a
        self.sigma_sq = sigma_sq

    def __call__(self, gradients):
        """Compute the kernel matrix.

        Parameters
        ----------
        gradients : RAS+b
            .

        Returns
        -------
        K : :obj:`~numpy.ndarray`, shape (n_samples, n_samples)
            Kernel matrix.
        """

        # ToDo
        # Call compute_squared_exponential_covariance_kernel
        pass

    def diag(self, X):
        """Return the diagonal of the kernel matrix.

        Parameters
        ----------
        X : :obj:`~numpy.ndarray`, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :obj:`~numpy.ndarray`, shape (n_samples,)
            Diagonal of the kernel matrix.
        """

        return np.full(X.shape[0], self.lambda_ + self.sigma_sq)

    def is_stationary(self):
        """Return whether the kernel is stationary.

        Returns
        -------
        :obj:`bool`
            Returns always ``True``.
        """

        return True

    def get_params(self, deep=True):
        """Get parameters of the kernel.

        Parameters
        ----------
        deep : :obj:`bool`
            Whether to return the parameters of the contained subobjects.

        Returns
        -------
        params : :obj:`dict`
            Parameter names mapped to their values.
        """

        return {"lambda_": self.lambda_, "a": self.a, "sigma_sq": self.sigma_sq}

    def set_params(self, **params):
        """Set parameters of the kernel.

        Parameters
        ----------
        params : :obj:`dict`
            Kernel parameters.

        Returns
        -------
        self
        """

        self.lambda_ = params.get("lambda_", self.lambda_)
        self.a = params.get("a", self.a)
        self.sigma_sq = params.get("sigma_sq", self.sigma_sq)
        return self


def compute_squared_exponential_shell_covariance(grpi, grpb, ell):
    r"""Compute the squared exponential smooth function describing how the
    covariance changes along the b direction.

    It uses the log of the b-values as the measure of distance along the
    b-direction according to eq. 15 in [Andersson15]_.

    .. math::

        C_{b}(b, b'; \ell) = \exp\left( - \frac{(\log b - \log b')^2}{2 \ell^2} \right)

    Parameters
    ----------
    grpi : :obj:`~numpy.ndarray`
        Group of indices.
    grpb : :obj:`~numpy.ndarray`
        Groups of b-values.
    ell : float

    Returns
    -------
        The squared exponential function.
    """

    # Compute log probability of b-values
    log_grpb = np.log(grpb)
    bv_diff = log_grpb[grpi[:, None]] - log_grpb[grpi]
    return np.exp(-(bv_diff**2) / (2 * ell**2))


def compute_squared_exponential_covariance_kernel(K, angle_mat, thpar, grpb, grpi):
    r"""Compute the squared exponential covariance matrix following to eq. 14 in
    [Andersson15]_.

    .. math::

        k(\textbf{x}, \textbf{x'}) = C_{\theta}(\mathbf{g}, \mathbf{g'}; a) C_{b}(\abs{b - b'}; \ell)

    where :math:`C_{\theta}` is given by:

    .. math::

        \begin{equation}
        C(\theta) =
        \begin{cases}  1 - \frac{3 \theta}{2 a} + \frac{\theta^3}{2 a^3} & \textnormal{if} \; \theta \leq a \\
        0 & \textnormal{if} \; \theta > a
        \end{cases}
        \end{equation}

   :math:`\theta` being computed as:

    .. math::

        \theta(\mathbf{g}, \mathbf{g'}) = \arccos(\abs{\langle \mathbf{g}, \mathbf{g'} \rangle})

    and :math:`C_{b}` is given by:

    .. math::

        C_{b}(b, b'; \ell) = \exp\left( - \frac{(\log b - \log b')^2}{2 \ell^2} \right)

    being :math:`b` and :math:`b'` the b-values, and :math:`\mathbf{g}` and
    :math:`\mathbf{g'}` the unit diffusion-encoding gradient unit vectors of the
    shells; and :math:`{a, \ell}` some hyperparameters.

    Parameters
    ----------

    Returns
    -------
    """

    sm = thpar[0]
    a = thpar[1]
    ell = thpar[2]

    # Compute angular covariance
    # ToDo
    # Vectorize this/take it from the single shell PR
    for j in range(K.shape[1]):
        for i in range(j, K.shape[0]):
            theta = angle_mat[i + 1, j + 1]
            if a > theta:
                K[i + 1, j + 1] = sm * (1.0 - 1.5 * theta / a + 0.5 * (theta**3) / (a**3))
            else:
                K[i + 1, j + 1] = 0.0

    # Compute b-value covariance
    # ToDo
    # Vectorize this/call compute_squared_exponential_shell_covariance
    if ngrp > 1:
        log_grpb = np.log(grpb())
        for j in range(K.shape[1]):
            for i in range(j + 1, K.shape[0]):
                bvdiff = log_grpb[grpi[i]] - log_grpb[grpi[j]]
                if bvdiff:
                    K[i + 1, j + 1] *= np.exp(-(bvdiff**2) / (2 * ell**2))
