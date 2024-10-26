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

from numbers import Integral, Real
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import optimize
from scipy.optimize._minimize import Bounds
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Hyperparameter,
    Kernel,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils._param_validation import Interval, StrOptions

BOUNDS_A: tuple[float, float] = (0.1, 0.5 * np.pi)
"""The limits for the parameter *a*."""
BOUNDS_LAMBDA: tuple[float, float] = (1.0, 1000)
"""The limits for the parameter lambda."""
THETA_EPSILON: float = 1e-5
"""Minimum nonzero angle."""
LBFGS_CONFIGURABLE_OPTIONS = {"disp", "maxiter", "ftol", "gtol"}
"""The set of extended options that can be set on the default BFGS."""
CONFIGURABLE_OPTIONS: Mapping[str, set] = {
    "Nelder-Mead": {"disp", "maxiter", "adaptive", "fatol"},
    "CG": {"disp", "maxiter", "gtol"},
}
"""
A mapping from optimizer names to the option set they allow.

Add new optimizers to this list, including what options may be
configured.
"""
NONGRADIENT_METHODS = {"Nelder-Mead"}
"""A set of gradients that do not allow analytical gradients."""
SUPPORTED_OPTIMIZERS = set(CONFIGURABLE_OPTIONS.keys()) | {"fmin_l_bfgs_b"}
"""A set of supported optimizers (automatically created)."""


class EddyMotionGPR(GaussianProcessRegressor):
    """
    A GP regressor specialized for eddymotion.

    This specialization of the default GP regressor is created to allow
    the following extended behaviors:

    * Pacify Scikit-learn's estimator parameter checker to allow optimizers
      given by name (as a string) other than the default BFGS.
    * Enable custom options of optimizers.
      See :obj:`~scipy.optimize.minimize` for the available options.
      Please note that only a few of them are currently supported.

    In the future, this specialization would be the right place for hyperparameter
    optimization using cross-validation and such.

    """

    _parameter_constraints: dict = {
        "kernel": [None, Kernel],
        "alpha": [Interval(Real, 0, None, closed="left"), np.ndarray],
        "optimizer": [StrOptions(SUPPORTED_OPTIMIZERS), callable, None],
        "n_restarts_optimizer": [Interval(Integral, 0, None, closed="left")],
        "copy_X_train": ["boolean"],
        "n_targets": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        kernel: Kernel | None = None,
        *,
        alpha: float = 1e-10,
        optimizer: str | Callable | None = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 0,
        copy_X_train: bool = True,
        n_targets: int | None = None,
        random_state: int | None = None,
        eval_gradient: bool = True,
        tol: float | None = None,
        disp: bool | int | None = None,
        maxiter: int | None = None,
        ftol: float | None = None,
        gtol: float | None = None,
        adaptive: bool | int | None = None,
        fatol: float | None = None,
    ):
        super().__init__(
            kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=False,  # We control normalization
            copy_X_train=copy_X_train,
            n_targets=n_targets,
            random_state=random_state,
        )

        self.tol = tol
        self.eval_gradient = eval_gradient if optimizer not in NONGRADIENT_METHODS else False
        self.maxiter = maxiter
        self.disp = disp
        self.ftol = ftol
        self.gtol = gtol
        self.adaptive = adaptive
        self.fatol = fatol

    def _constrained_optimization(
        self,
        obj_func: Callable,
        initial_theta: np.ndarray,
        bounds: Sequence[tuple[float, float]] | Bounds,
    ) -> tuple[float, float]:
        options = {}
        if self.optimizer == "fmin_l_bfgs_b":
            from sklearn.utils.optimize import _check_optimize_result

            for name in LBFGS_CONFIGURABLE_OPTIONS:
                if (value := getattr(self, name, None)) is not None:
                    options[name] = value

            opt_res = optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                bounds=bounds,
                jac=self.eval_gradient,
                options=options,
                args=(self.eval_gradient,),
                tol=self.tol,
            )
            _check_optimize_result("lbfgs", opt_res)
            return opt_res.x, opt_res.fun

        if isinstance(self.optimizer, str) and self.optimizer in CONFIGURABLE_OPTIONS:
            for name in CONFIGURABLE_OPTIONS[self.optimizer]:
                if (value := getattr(self, name, None)) is not None:
                    options[name] = value

            opt_res = optimize.minimize(
                obj_func,
                initial_theta,
                method=self.optimizer,
                bounds=bounds,
                jac=self.eval_gradient,
                options=options,
                args=(self.eval_gradient,),
                tol=self.tol,
            )
            return opt_res.x, opt_res.fun

        if callable(self.optimizer):
            return self.optimizer(obj_func, initial_theta, bounds=bounds)

        raise ValueError(f"Unknown optimizer {self.optimizer}.")

    def fit(self, X, y):
        r"""Fit Gaussian process regression model.

        This overshadows Scikit-Learn's implementation to remove access over
        the normalization of y.

        Instead, and following Anderson 2015 and what it looks like from
        `FSL's souce code <https://git.fmrib.ox.ac.uk/fsl/eddy/-/blob/\
2480dda293d4cec83014454db3a193b87921f6b0/DiffusionGP.cpp#L218>`__,
        our fit method should also remove the mean.

        From their paper (p. 167, end of first column):

            Typically one just substracts the mean (:math:`\hat{\mathbf{f}}`)
            from :math:`\mathbf{f}` and then add it back to
            :math:`f^{*}`, which is analogous to what is often done in
            "traditional" regression.

        It is still unclear to me how :math:`f^{*}` is interpolated from
        :math:`\hat{\mathbf{f}}` (grand mean?), or whether the fitted model
        accounts for that factor (I doubt it).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        :obj:`~sklearn.gaussian_process.GaussianProcessRegressor`
            GaussianProcessRegressor class instance.
        """
        super().fit(X, y - y.mean(1)[:, None])

        return self


class ExponentialKriging(Kernel):
    """A scikit-learn's kernel for DWI signals."""

    def __init__(
        self,
        beta_a: float = 0.01,
        beta_l: float = 2.0,
        a_bounds: tuple[float, float] = BOUNDS_A,
        l_bounds: tuple[float, float] = BOUNDS_LAMBDA,
    ):
        r"""
        Initialize an exponential Kriging kernel.

        Parameters
        ----------
        beta_a : :obj:`float`, optional
            Minimum angle in rads.
        beta_l : :obj:`float`, optional
            The :math:`\lambda` hyperparameter.
        a_bounds : :obj:`tuple`, optional
            Bounds for the a parameter.
        l_bounds : :obj:`tuple`, optional
            Bounds for the :math:`\lambda` hyperparameter.

        """
        self.beta_a = beta_a
        self.beta_l = beta_l
        self.a_bounds = a_bounds
        self.l_bounds = l_bounds

    @property
    def hyperparameter_a(self) -> Hyperparameter:
        return Hyperparameter("beta_a", "numeric", self.a_bounds)

    @property
    def hyperparameter_beta_l(self) -> Hyperparameter:
        return Hyperparameter("beta_l", "numeric", self.l_bounds)

    def __call__(
        self, X: np.ndarray, Y: np.ndarray | None = None, eval_gradient: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Return the kernel K(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : :obj:`~numpy.ndarray`
            Gradient table (X)
        Y : :obj:`~numpy.ndarray`, optional
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
        C_theta = exponential_covariance(thetas, self.beta_a)

        if not eval_gradient:
            return self.beta_l * C_theta

        K_gradient = np.zeros((*thetas.shape, 2))
        K_gradient[..., 0] = self.beta_l * C_theta * thetas / self.beta_a**2  # Derivative w.r.t. a
        K_gradient[..., 1] = C_theta

        return self.beta_l * C_theta, K_gradient

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
        return self.beta_l * np.ones(X.shape[0])

    def is_stationary(self) -> bool:
        """Returns whether the kernel is stationary."""
        return True

    def __repr__(self) -> str:
        return f"ExponentialKriging (a={self.beta_a}, λ={self.beta_l})"


class SphericalKriging(Kernel):
    """A scikit-learn's kernel for DWI signals."""

    def __init__(
        self,
        beta_a: float = 0.01,
        beta_l: float = 2.0,
        a_bounds: tuple[float, float] = BOUNDS_A,
        l_bounds: tuple[float, float] = BOUNDS_LAMBDA,
    ):
        r"""
        Initialize a spherical Kriging kernel.

        Parameters
        ----------
        beta_a : :obj:`float`, optional
            Minimum angle in rads.
        beta_l : :obj:`float`, optional
            The :math:`\lambda` hyperparameter.
        a_bounds : :obj:`tuple`, optional
            Bounds for the ``a`` parameter.
        l_bounds : :obj:`tuple`, optional
            Bounds for the :math:`\lambda` hyperparameter.

        """
        self.beta_a = beta_a
        self.beta_l = beta_l
        self.a_bounds = a_bounds
        self.l_bounds = l_bounds

    @property
    def hyperparameter_a(self) -> Hyperparameter:
        return Hyperparameter("beta_a", "numeric", self.a_bounds)

    @property
    def hyperparameter_beta_l(self) -> Hyperparameter:
        return Hyperparameter("beta_l", "numeric", self.l_bounds)

    def __call__(
        self, X: np.ndarray, Y: np.ndarray | None = None, eval_gradient: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Return the kernel K(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : :obj:`~numpy.ndarray`
            Gradient table (X)
        Y : :obj:`~numpy.ndarray`, optional
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
            hyperparameter of the kernel. Only returned when ``eval_gradient``
            is True.

        """
        thetas = compute_pairwise_angles(X, Y)
        C_theta = spherical_covariance(thetas, self.beta_a)

        if not eval_gradient:
            return self.beta_l * C_theta

        deriv_a = np.zeros_like(thetas)
        nonzero = thetas <= self.beta_a
        deriv_a[nonzero] = (
            1.5
            * self.beta_l
            * (thetas[nonzero] / self.beta_a**2 - thetas[nonzero] ** 3 / self.beta_a**4)
        )
        K_gradient = np.dstack((deriv_a, C_theta))

        return self.beta_l * C_theta, K_gradient

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
        return self.beta_l * np.ones(X.shape[0])

    def is_stationary(self) -> bool:
        """Returns whether the kernel is stationary."""
        return True

    def __repr__(self) -> str:
        return f"SphericalKriging (a={self.beta_a}, λ={self.beta_l})"


def exponential_covariance(theta: np.ndarray, a: float) -> np.ndarray:
    """
    Compute the exponential covariance for given distances and scale parameter.

    Parameters
    ----------
    theta : :obj:`~numpy.ndarray`
        Array of distances between points.
    a : :obj:`float`
        Scale parameter that controls the range of the covariance function.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Exponential covariance values for the input distances.

    """
    return np.exp(-theta / a)


def spherical_covariance(theta: np.ndarray, a: float) -> np.ndarray:
    """
    Compute the spherical covariance for given distances and scale parameter.

    Parameters
    ----------
    theta : :obj:`~numpy.ndarray`
        Array of distances between points.
    a : :obj:`float`
        Scale parameter that controls the range of the covariance function.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Spherical covariance values for the input distances.

    """
    return np.where(theta <= a, 1 - 1.5 * theta / a + 0.5 * (theta**3) / (a**3), 0.0)


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
    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), optional
        Input data. If ``None``, the output will be the pairwise
        similarities between all samples in ``X``.
    dense_output : :obj:`bool`, optional
        Whether to return dense output even when the input is sparse. If
        ``False``, the output is sparse if both input arrays are sparse.
    closest_polarity : :obj:`bool`, optional
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
    thetas = np.arccos(np.abs(cosines)) if closest_polarity else np.arccos(cosines)
    thetas[np.abs(thetas) < THETA_EPSILON] = 0.0
    return thetas
