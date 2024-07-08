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
"""DIPY-like models (a sandbox to trial them out before upstreaming to DIPY)."""

from __future__ import annotations

from sys import modules

import numpy as np
from dipy.core.gradients import GradientTable
from dipy.reconst.base import ReconstModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct,
    Hyperparameter,
    Kernel,
    WhiteKernel,
)


def gp_prediction(
    model_gtab: GradientTable,
    gtab: GradientTable | np.ndarray,
    model: GaussianProcessRegressor,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Predicts one or more DWI orientations given a model.

    This function checks if the model is fitted and then extracts
    orientations and potentially b-values from the gtab. It predicts the mean
    and standard deviation of the DWI signal using the model.

    Parameters
    ----------
    model: :obj:`~sklearn.gaussian_process.GaussianProcessRegressor`
        A fitted GaussianProcessRegressor model.
    gtab: :obj:`~dipy.core.gradients.GradientTable`
        Gradient table with one or more orientations at which the GP will be evaluated.
    mask: :obj:`numpy.ndarray`
        A boolean mask indicating which voxels to use (optional).

    Returns
    -------
    :obj:`numpy.ndarray`
        A 3D or 4D array with the simulated gradient(s).

    """

    # Check it's fitted as they do in sklearn internally
    # https://github.com/scikit-learn/scikit-learn/blob/972e17fe1aa12d481b120ad4a3dc076bae736931/\
    # sklearn/gaussian_process/_gpr.py#L410C9-L410C42
    if not hasattr(model, "X_train_"):
        raise RuntimeError("Model is not yet fitted.")

    # Extract orientations from gtab, and highly likely, the b-value too.
    return model.predict(gtab, return_std=False)


class GaussianProcessModel(ReconstModel):
    """A Gaussian Process (GP) model to simulate single- and multi-shell DWI data."""

    __slots__ = (
        "kernel",
        "_modelfit",
    )

    def __init__(
        self,
        kernel_model: str = "spherical",
        lambda_s: float = 2.0,
        a: float = 0.1,
        sigma_sq: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        """A GP-based DWI model [Andersson15]_.

        Parameters
        ----------
        kernel_model : :obj:`str`
            Kernel model to calculate the GP's covariance matrix.

        References
        ----------
        .. [Andersson15] Jesper L.R. Andersson and Stamatios N. Sotiropoulos.
           Non-parametric representation and prediction of single- and multi-shell
           diffusion-weighted MRI data using Gaussian processes. NeuroImage, 122:166-176, 2015.
           doi:\
           `10.1016/j.neuroimage.2015.07.067 <https://doi.org/10.1016/j.neuroimage.2015.07.067>`__.

        """

        ReconstModel.__init__(self, None)
        self.kernel = (
            PairwiseOrientationKernel(
                weighting=kernel_model,
                a=a,
                lambda_s=lambda_s,
                sigma_sq=sigma_sq,
            )
            if kernel_model != "test"
            else DotProduct() + WhiteKernel()
        )

    def fit(
        self,
        data: np.ndarray,
        gtab: GradientTable | None = None,
        mask: np.ndarray[bool] | None = None,
        random_state: int = 0,
    ) -> GPFit:
        """Fit method of the DTI model class

        Parameters
        ----------
        data : :obj:`~numpy.ndarray`
            The measured signal from one voxel.
        gtab : :obj:`~dipy.core.gradients.GradientTable`
            The gradient table corresponding to the training data.
        mask : :obj:`~numpy.ndarray`
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]

        Returns
        -------
        :obj:`~eddymotion.model.dipy.GPFit`
            A model fit container object.

        """

        y = (
            data[mask[..., None]] if mask is not None else np.reshape(data, (-1, data.shape[-1]))
        ).T

        # sklearn wants (n_samples, n_features) as X's shape
        X = gtab.bvecs

        # sklearn wants (n_samples, n_targets) for Y, where n_targets = n_voxels to simulate.
        if (signal_dirs := y.shape[0]) != (grad_dirs := X.shape[0]):
            raise ValueError(
                f"Mismatched data {signal_dirs} and gradient table {grad_dirs} sizes."
            )

        gpr = GaussianProcessRegressor(
            kernel=self.kernel,
            random_state=random_state,
            n_targets=y.shape[1],
        )
        self._modelfit = GPFit(
            gtab=gtab,
            model=gpr.fit(X, y),
            mask=mask,
        )
        return self._modelfit

    # @multi_voxel_fit
    # def multi_fit(self, data_thres, mask=None, **kwargs):
    #     return GPFit(gpr.fit(self.gtab, data_thres))

    def predict(
        self,
        gtab: GradientTable,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict using the Gaussian process model of the DWI signal for one or more gradients.

        Parameters
        ----------
        gtab : :obj:`~dipy.core.gradients.GradientTable`
            One or more gradient orientations at which the GP will be evaluated.

        Returns
        -------
        :obj:`numpy.ndarray`
            A 3D or 4D array with the simulated gradient(s).

        """
        return self._modelfit.predict(gtab)


class GPFit:
    """
    A container class to store the fitted Gaussian process model and mask information.

    This class is typically returned by the `fit` and `multi_fit` methods of the
    `GaussianProcessModel` class. It holds the fitted model and the mask used during fitting.

    Attributes
    ----------
    fitted_gtab : :obj:`~dipy.core.gradients.GradientTable`
        The original gradient table with which the GP has been fitted.
    model: :obj:`~sklearn.gaussian_process.GaussianProcessRegressor`
        The fitted Gaussian process regressor object.
    mask: :obj:`~numpy.ndarray`
        The boolean mask used during fitting (can be ``None``).

    """

    def __init__(
        self,
        gtab: GradientTable,
        model: GaussianProcessRegressor,
        mask: np.ndarray | None = None,
    ) -> None:
        """
        Initialize a Gaussian Process fit container.

        Parameters
        ----------
        gtab : :obj:`~dipy.core.gradients.GradientTable`
            The gradient table with which the GP has been fitted.
        model: :obj:`~sklearn.gaussian_process.GaussianProcessRegressor`
            The fitted Gaussian process regressor object.
        mask: :obj:`~numpy.ndarray`
            The boolean mask used during fitting (can be ``None``).

        """
        self.fitted_gtab = gtab
        self.model = model
        self.mask = mask

    def predict(
        self,
        gtab: GradientTable,
    ) -> np.ndarray:
        """
        Generate DWI signal based on a fitted Gaussian Process.

        Parameters
        ----------
        gtab: :obj:`~dipy.core.gradients.GradientTable`
            Gradient table with one or more orientations at which the GP will be evaluated.

        Returns
        -------
        :obj:`numpy.ndarray`
            A 3D or 4D array with the simulated gradient(s).

        """
        return gp_prediction(self.fitted_gtab, gtab, self.model, mask=self.mask)


def _ensure_positive_scale(
    a: float,
) -> None:
    if a <= 0:
        raise ValueError(f"a must be strictly positive. Provided: {a}")


def compute_exponential_covariance(
    theta: np.ndarray,
    a: float,
) -> np.ndarray:
    r"""Compute the exponential covariance matrix following eq. 9 in [Andersson15]_.

    .. math::

        C(\theta) = \exp(- \frac{\theta}{a})

    Parameters
    ----------
    theta : :obj:`~numpy.ndarray`
        Pairwise angles across diffusion gradient encoding directions.
    a : :obj:`float`
        Positive scale parameter that here determines the "distance" at which θ
        the covariance goes to zero.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Exponential covariance function values.

    """

    _ensure_positive_scale(a)

    return np.exp(-theta / a)


def compute_spherical_covariance(
    theta: np.ndarray,
    a: float,
) -> np.ndarray:
    r"""Compute the spherical covariance matrix following eq. 10 in [Andersson15]_.

    .. math::

        C(\theta) = \begin{cases}
            1 - \frac{3 \theta}{2 a} + \frac{\theta^3}{2 a^3} & \textnormal{if} \; \theta \leq a \\
            0 & \textnormal{if} \; \theta > a
        \end{cases}

    Parameters
    ----------
    theta : :obj:`~numpy.ndarray`
        Pairwise angles across diffusion gradient encoding directions.
    a : :obj:`float`
        Positive scale parameter that here determines the "distance" at which θ
        the covariance goes to zero.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Spherical covariance matrix.

    """

    _ensure_positive_scale(a)

    return np.where(theta <= a, 1 - 3 * (theta / a) ** 2 + 2 * (theta / a) ** 3, 0)


def compute_derivative(
    theta: np.ndarray,
    kernel: np.ndarray,
    weighting: str,
    params: dict[float],
):
    """
    Compute the analytical derivative of the kernel.

    Parameters
    ----------
    theta : :obj:`~numpy.ndarray`
        Pairwise angles across diffusion gradient encoding directions.
    kernel : :obj:`~numpy.ndarray`
        Current kernel.
    weighting : :obj:`str`
        The kind of kernel which derivatives will be calculated.
    params : :obj:`dict`
        Current parameter values.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Gradients of the kernel.

    """

    n_partials = len(params)

    a = params.pop("a")
    lambda_s = params.pop("lambda_s")

    min_angles = theta > a

    if weighting == "spherical":
        deriv_a = (3 * theta[min_angles] / a**2) - (1.5 * (theta[min_angles] / a) ** 2) / a
    elif weighting == "exponential":
        deriv_a = theta[min_angles] * a * np.exp(-theta[min_angles] / a)
    else:
        raise ValueError(f"Unknown kernel weighting '{weighting}'.")

    K_gradient = np.zeros((*theta.shape, n_partials))
    K_gradient[..., 0] = kernel / lambda_s
    K_gradient[..., 1][min_angles] = lambda_s * deriv_a
    K_gradient[..., 2][theta > 1e-5] = 2

    return K_gradient


def compute_pairwise_angles(
    gtab_X: GradientTable | np.ndarray,
    gtab_Y: GradientTable | np.ndarray | None = None,
    closest_polarity: bool = True,
) -> np.ndarray:
    r"""Compute pairwise angles across diffusion gradient encoding directions.

    Following [Andersson15]_, it computes the smallest of the angles between
    each pair if ``closest_polarity`` is ``True``, i.e.

    .. math::

        \theta(\mathbf{g}, \mathbf{g'}) = \arccos(\abs{\langle \mathbf{g}, \mathbf{g'} \rangle})

    Parameters
    ----------
    gtab_X: :obj:`~dipy.core.gradients.GradientTable`
        Gradient table
    gtab_Y: :obj:`~dipy.core.gradients.GradientTable`
        Gradient table
    closest_polarity : :obj:`bool`
        ``True`` to consider the smallest of the two angles between the crossing
         lines resulting from reversing each vector pair.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Pairwise angles across diffusion gradient encoding directions.

    Examples
    --------
    >>> from dipy.core.gradients import gradient_table
    >>> bvecs = np.asarray([(1.0, -1.0), (0.0, 0.0), (0.0, 0.0)])
    >>> gtab = gradient_table([1000] * bvecs.shape[-1], bvecs)
    >>> compute_pairwise_angles(gtab, closest_polarity=False)[0, 1]  # doctest: +ELLIPSIS
    3.1415...
    >>> bvecs1 = np.asarray([(1.0, -1.0), (0.0, 0.0), (0.0, 0.0)])
    >>> bvecs2 = np.asarray([(1.0, -1.0), (0.0, 0.0), (0.0, 0.0)])
    >>> gtab1 = gradient_table([1000] * bvecs1.shape[-1], bvecs1)
    >>> gtab2 = gradient_table([1000] * bvecs2.shape[-1], bvecs2)
    >>> compute_pairwise_angles(gtab1, gtab2, closest_polarity=False)[0, 1]  # doctest: +ELLIPSIS
    3.1415...
    >>> bvecs = np.asarray([(1.0, -1.0), (0.0, 0.0), (0.0, 0.0)])
    >>> gtab = gradient_table([1000] * bvecs.shape[-1], bvecs)
    >>> compute_pairwise_angles(gtab)[0, 1]
    0.0

    References
    ----------
    .. [Andersson15] J. L. R. Andersson. et al., An integrated approach to
       correction for off-resonance effects and subject movement in diffusion MR
       imaging, NeuroImage 125 (2016) 1063–1078

    """

    bvecs_X = getattr(gtab_X, "bvecs", gtab_X)
    bvecs_X = np.array(bvecs_X.T) / np.linalg.norm(bvecs_X, axis=1)

    if gtab_Y is None:
        bvecs_Y = bvecs_X
    else:
        bvecs_Y = np.array(getattr(gtab_Y, "bvecs", gtab_Y))
        if bvecs_Y.ndim == 1:
            bvecs_Y = bvecs_Y[np.newaxis, ...]
        bvecs_Y = bvecs_Y.T / np.linalg.norm(bvecs_Y, axis=1)

    cosines = np.clip(bvecs_X.T @ bvecs_Y, -1.0, 1.0)
    return np.arccos(np.abs(cosines) if closest_polarity else cosines)


class PairwiseOrientationKernel(Kernel):
    """A scikit-learn's kernel for DWI signals."""

    def __init__(
        self,
        weighting: str = "exponential",
        lambda_s: float = 2.0,
        a: float = 0.1,
        sigma_sq: float = 1.0,
        lambda_s_bounds: tuple[float, float] = (1e-5, 1e4),
        a_bounds: tuple[float, float] = (1e-5, np.pi),
        sigma_sq_bounds: tuple[float, float] = (1e-5, 1e4),
    ):
        self._weighting = weighting  # For __repr__
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

    def __call__(self, gtab_X, gtab_Y=None, eval_gradient=False):
        """
        Return the kernel K(gtab_X, gtab_Y) and optionally its gradient.

        Parameters
        ----------
        gtab_X: :obj:`~dipy.core.gradients.GradientTable`
            Gradient table (X)
        gtab_Y: :obj:`~dipy.core.gradients.GradientTable`
            Gradient table (Y, optional)
        eval_gradient : :obj:`bool`
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when gtab_Y is ``None``.

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
        thetas = compute_pairwise_angles(gtab_X, gtab_Y)
        collinear = np.abs(thetas) < 1e-5
        thetas[collinear] = 0.0

        covfunc = getattr(modules[__name__], f"compute_{self._weighting}_covariance")

        K = self.lambda_s * covfunc(thetas, self.a)
        K[collinear] += self.sigma_sq

        if not eval_gradient:
            return K

        if gtab_Y is not None:
            raise RuntimeError("Gradients should not be calculated in inference time")

        K_gradient = compute_derivative(
            thetas,
            K,
            weighting=self._weighting,
            params=self.get_params(),
        )

        return K, K_gradient

    def diag(self, X):
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
        try:
            n = len(X)
        except TypeError:
            n = len(X.bvals)

        covfunc = getattr(modules[__name__], f"compute_{self._weighting}_covariance")
        return self.lambda_s * covfunc(np.zeros(n), self.a) + self.sigma_sq

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return True

    def __repr__(self):
        return (
            f"{self._weighting} kernel"
            f"(a={self.a}, lambda_s={self.lambda_s}, sigma_sq={self.sigma_sq})"
        )

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
