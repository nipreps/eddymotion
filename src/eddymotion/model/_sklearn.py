# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# © The NiPreps Developers <nipreps@gmail.com>
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
r"""
Derivations from scikit-learn for Gaussian Processes.

Gaussian Process Model: Pairwise orientation angles
---------------------------------------------------
Squared Exponential covariance kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Kernel based on a squared exponential function for Gaussian processes on
multi-shell DWI data following to eqs. 14 and 16 in [Andersson15]_.
For a 2-shell case, the :math:`\mathbf{K}` kernel can be written as:

.. math::
    \begin{equation}
    \mathbf{K} = \left[
    \begin{matrix}
        \lambda C_{\theta}(\theta (\mathbf{G}_{1}); a) + \sigma_{1}^{2} \mathbf{I} &
        \lambda C_{\theta}(\theta (\mathbf{G}_{2}, \mathbf{G}_{1}); a) C_{b}(b_{2}, b_{1}; \ell) \\
        \lambda C_{\theta}(\theta (\mathbf{G}_{1}, \mathbf{G}_{2}); a) C_{b}(b_{1}, b_{2}; \ell) &
        \lambda C_{\theta}(\theta (\mathbf{G}_{2}); a) + \sigma_{2}^{2} \mathbf{I} \\
    \end{matrix}
    \right]
    \end{equation}

**Squared exponential shell-wise covariance kernel**:
Compute the squared exponential smooth function describing how the
covariance changes along the b direction.
It uses the log of the b-values as the measure of distance along the
b-direction according to eq. 15 in [Andersson15]_.

.. math::
    C_{b}(b, b'; \ell) = \exp\left( - \frac{(\log b - \log b')^2}{2 \ell^2} \right).

**Squared exponential covariance kernel**:
Compute the squared exponential covariance matrix following to eq. 14 in [Andersson15]_.

.. math::
    k(\textbf{x}, \textbf{x'}) = C_{\theta}(\mathbf{g}, \mathbf{g'}; a) C_{b}(|b - b'|; \ell)

where :math:`C_{\theta}` is given by:

.. math::
    \begin{equation}
    C(\theta) =
    \begin{cases}
    1 - \frac{3 \theta}{2 a} + \frac{\theta^3}{2 a^3} & \textnormal{if} \; \theta \leq a \\
    0 & \textnormal{if} \; \theta > a
    \end{cases}
    \end{equation}

:math:`\theta` being computed as:

.. math::
    \theta(\mathbf{g}, \mathbf{g'}) = \arccos(|\langle \mathbf{g}, \mathbf{g'} \rangle|)

and :math:`C_{b}` is given by:

.. math::
    C_{b}(b, b'; \ell) = \exp\left( - \frac{(\log b - \log b')^2}{2 \ell^2} \right)

:math:`b` and :math:`b'` being the b-values, and :math:`\mathbf{g}` and
:math:`\mathbf{g'}` the unit diffusion-encoding gradient unit vectors of the
shells; and :math:`{a, \ell}` some hyperparameters.

"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import optimize
from scipy.optimize._minimize import Bounds
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Hyperparameter,
    Kernel,
)
from sklearn.metrics.pairwise import cosine_similarity

BOUNDS_A: tuple[float, float] = (1e-4, np.pi)
BOUNDS_LAMBDA_S: tuple[float, float] = (1e-4, 1e4)
THETA_EPSILON: float = 1e-5


class EddyMotionGPR(GaussianProcessRegressor):
    def __init__(
        self,
        kernel: Kernel | None = None,
        *,
        alpha: float = 1e-10,
        optimizer: str | Callable | None = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 0,
        normalize_y: bool = True,
        copy_X_train: bool = True,
        n_targets: int | None = None,
        random_state: int | None = None,
        max_iter: int = 2e05,
        gtol: float = 1e-06,
    ):
        self.max_iter = max_iter
        self.gtol = gtol

        super().__init__(
            kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            n_targets=n_targets,
            random_state=random_state,
        )

    def _constrained_optimization(
        self,
        obj_func: Callable,
        initial_theta: np.ndarray,
        bounds: Sequence[tuple[float, float]] | Bounds,
    ) -> tuple[float, float]:
        from sklearn.utils.optimize import _check_optimize_result

        opt_res = optimize.minimize(
            obj_func,
            initial_theta,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": self.max_iter, "gtol": self.gtol},
        )
        _check_optimize_result("lbfgs", opt_res)
        return opt_res.x, opt_res.fun


class ExponentialKriging(Kernel):
    """A scikit-learn's kernel for DWI signals."""

    def __init__(
        self,
        a: float = 0.01,
        lambda_s: float = 2.0,
        a_bounds: tuple[float, float] = BOUNDS_A,
        lambda_s_bounds: tuple[float, float] = BOUNDS_LAMBDA_S,
    ):
        r"""
        Initialize an exponential Kriging kernel.

        Parameters
        ----------
        a : :obj:`float`
            Minimum angle in rads.
        lambda_s : :obj:`float`
            The :math:`\lambda_s` hyperparameter.
        a_bounds : :obj:`tuple`
            Bounds for the a parameter.
        lambda_s_bounds : :obj:`tuple`
            Bounds for the :math:`\lambda_s` hyperparameter.

        """
        self.a = a
        self.lambda_s = lambda_s
        self.a_bounds = a_bounds
        self.lambda_s_bounds = lambda_s_bounds

    @property
    def hyperparameter_a(self) -> Hyperparameter:
        return Hyperparameter("a", "numeric", self.a_bounds)

    @property
    def hyperparameter_lambda_s(self) -> Hyperparameter:
        return Hyperparameter("lambda_s", "numeric", self.lambda_s_bounds)

    def __call__(
        self, X: np.ndarray, Y: np.ndarray | None = None, eval_gradient: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Return the kernel K(X, Y) and optionally its gradient.

        Parameters
        ----------
        X: :obj:`~numpy.ndarray`
            Gradient table (X)
        Y: :obj:`~numpy.ndarray`, optional
            Gradient table (Y, optional)
        eval_gradient : :obj:`bool`, optional
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is ``None``.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.

        """
        thetas = compute_pairwise_angles(X, Y)
        thetas[np.abs(thetas) < THETA_EPSILON] = 0.0
        C_theta = np.exp(-thetas / self.a)

        if not eval_gradient:
            return self.lambda_s * C_theta

        K_gradient = np.zeros((*thetas.shape, 2))
        K_gradient[..., 0] = self.lambda_s * C_theta * thetas / self.a**2  # Derivative w.r.t. a
        K_gradient[..., 1] = C_theta

        return self.lambda_s * C_theta, K_gradient

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.lambda_s * np.ones(X.shape[0])

    def is_stationary(self) -> bool:
        """Returns whether the kernel is stationary."""
        return True

    def __repr__(self) -> str:
        return f"ExponentialKriging (a={self.a}, λₛ={self.lambda_s})"


class SphericalKriging(Kernel):
    """A scikit-learn's kernel for DWI signals."""

    def __init__(
        self,
        a: float = 0.01,
        lambda_s: float = 2.0,
        a_bounds: tuple[float, float] = BOUNDS_A,
        lambda_s_bounds: tuple[float, float] = BOUNDS_LAMBDA_S,
    ):
        r"""
        Initialize a spherical Kriging kernel.

        Parameters
        ----------
        a : :obj:`float`
            Minimum angle in rads.
        lambda_s : :obj:`float`
            The :math:`\lambda_s` hyperparameter.
        a_bounds : :obj:`tuple`
            Bounds for the a parameter.
        lambda_s_bounds : :obj:`tuple`
            Bounds for the :math:`\lambda_s` hyperparameter.

        """
        self.a = a
        self.lambda_s = lambda_s
        self.a_bounds = a_bounds
        self.lambda_s_bounds = lambda_s_bounds

    @property
    def hyperparameter_a(self) -> Hyperparameter:
        return Hyperparameter("a", "numeric", self.a_bounds)

    @property
    def hyperparameter_lambda_s(self) -> Hyperparameter:
        return Hyperparameter("lambda_s", "numeric", self.lambda_s_bounds)

    def __call__(
        self, X: np.ndarray, Y: np.ndarray | None = None, eval_gradient: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Return the kernel K(X, Y) and optionally its gradient.

        Parameters
        ----------
        X: :obj:`~numpy.ndarray`
            Gradient table (X)
        Y: :obj:`~numpy.ndarray`, optional
            Gradient table (Y, optional)
        eval_gradient : :obj:`bool`, optional
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is ``None``.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.

        """
        thetas = compute_pairwise_angles(X, Y)
        thetas[np.abs(thetas) < THETA_EPSILON] = 0.0

        nonzero = thetas <= self.a

        C_theta = np.zeros_like(thetas)
        C_theta[nonzero] = (
            1 - 1.5 * thetas[nonzero] / self.a + 0.5 * thetas[nonzero] ** 3 / self.a**3
        )

        if not eval_gradient:
            return self.lambda_s * C_theta

        deriv_a = np.zeros_like(thetas)
        deriv_a[nonzero] = (
            1.5 * self.lambda_s * (thetas[nonzero] / self.a**2 - thetas[nonzero] ** 3 / self.a**4)
        )
        K_gradient = np.dstack((deriv_a, C_theta))

        return self.lambda_s * C_theta, K_gradient

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.lambda_s * np.ones(X.shape[0])

    def is_stationary(self) -> bool:
        """Returns whether the kernel is stationary."""
        return True

    def __repr__(self) -> str:
        return f"SphericalKriging (a={self.a}, λₛ={self.lambda_s})"


def compute_pairwise_angles(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    closest_polarity: bool = True,
    dense_output: bool = True,
) -> np.ndarray:
    r"""Compute pairwise angles across diffusion gradient encoding directions.

    Following [Andersson15]_, it computes the smallest of the angles between
    each pair if ``closest_polarity`` is ``True``, i.e.,

    .. math::

        \theta(\mathbf{g}, \mathbf{g'}) = \arccos(|\langle \mathbf{g}, \mathbf{g'} \rangle|)

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        Input data.
    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), \
            default=None
        Input data. If ``None``, the output will be the pairwise
        similarities between all samples in ``X``.
    dense_output : :obj:`bool`, default=True
        Whether to return dense output even when the input is sparse. If
        ``False``, the output is sparse if both input arrays are sparse.
    closest_polarity : :obj:`bool`, default=True
        ``True`` to consider the smallest of the two angles between the crossing
         lines resulting from reversing each vector pair.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Pairwise angles across diffusion gradient encoding directions.

    Examples
    --------
    >>> X = np.asarray([(1.0, -1.0), (0.0, 0.0), (0.0, 0.0)]).T
    >>> compute_pairwise_angles(X, closest_polarity=False)[0, 1]  # doctest: +ELLIPSIS
    3.1415...
    >>> X = np.asarray([(1.0, -1.0), (0.0, 0.0), (0.0, 0.0)]).T
    >>> Y = np.asarray([(1.0, -1.0), (0.0, 0.0), (0.0, 0.0)]).T
    >>> compute_pairwise_angles(X, Y, closest_polarity=False)[0, 1]  # doctest: +ELLIPSIS
    3.1415...
    >>> X = np.asarray([(1.0, -1.0), (0.0, 0.0), (0.0, 0.0)]).T
    >>> compute_pairwise_angles(X)[0, 1]
    0.0

    References
    ----------
    .. [Andersson15] J. L. R. Andersson. et al., An integrated approach to
       correction for off-resonance effects and subject movement in diffusion MR
       imaging, NeuroImage 125 (2016) 1063-11078

    """

    cosines = np.clip(cosine_similarity(X, Y, dense_output=dense_output), -1.0, 1.0)
    return np.arccos(np.abs(cosines)) if closest_polarity else np.arccos(cosines)
