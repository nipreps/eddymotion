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

import numpy as np
from dipy.core.gradients import GradientTable
from dipy.reconst.base import ReconstModel
from sklearn.gaussian_process import GaussianProcessRegressor

# from dipy.reconst.multi_voxel import multi_voxel_fit


def gp_prediction(
    model_gtab: GradientTable,
    gtab: GradientTable,
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
    if not hasattr(model._gpr, "X_train_"):
        raise RuntimeError("Model is not yet fitted.")

    # Extract orientations from gtab, and highly likely, the b-value too.
    return model._gpr.predict(gtab, return_std=False)


def get_kernel(
    kernel_model: str,
    gtab: GradientTable | None = None,
) -> GaussianProcessRegressor.kernel:
    """
    Returns a Gaussian process kernel based on the provided string.

    Currently supports 'test' kernel which is a combination of DotProduct and WhiteKernel
    from scikit-learn. Raises a TypeError for unknown kernel models.

    Parameters
    ----------
    kernel_model: :obj:`str`
        The string representing the desired kernel model.

    Returns
    -------
    :obj:`GaussianProcessRegressor.kernel`
        A GaussianProcessRegressor kernel object.

    Raises
    ------
    TypeError: If the provided ``kernel_model`` is not supported.

    """

    if kernel_model == "spherical":
        raise NotImplementedError("Spherical kernel is not currently implemented.")

    if kernel_model == "exponential":
        raise NotImplementedError("Exponential kernel is not currently implemented.")

    if kernel_model == "test":
        from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

        return DotProduct() + WhiteKernel()

    raise TypeError(f"Unknown kernel '{kernel_model}'.")


class GaussianProcessModel(ReconstModel):
    """A Gaussian Process (GP) model to simulate single- and multi-shell DWI data."""

    __slots__ = (
        "kernel_model",
        "_modelfit",
    )

    def __init__(
        self,
        kernel_model: str = "spherical",
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
        self.kernel_model = kernel_model

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

        data = (
            data[mask[..., None]] if mask is not None else np.reshape(data, (-1, data.shape[-1]))
        )

        if data.shape[-1] != len(gtab):
            raise ValueError(
                f"Mismatched data {data.shape[-1]} and gradient table {len(gtab)} sizes."
            )

        gpr = GaussianProcessRegressor(
            kernel=get_kernel(self.kernel_model, gtab=gtab),
            random_state=random_state,
        )
        self._modelfit = GPFit(
            gpr.fit(gtab.gradients, data),
            gtab=gtab,
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


def compute_pairwise_angles(
    gtab_X: GradientTable,
    gtab_Y: GradientTable | None = None,
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
    >>> compute_pairwise_angles(
    ...     ((1.0, -1.0), (0.0, 0.0), (0.0, 0.0)),
    ...     closest_polarity=False,
    ... )[0, 1]  # doctest: +ELLIPSIS
    3.1415...
    >>> compute_pairwise_angles(
    ...     ((1.0, -1.0), (0.0, 0.0), (0.0, 0.0)),
    ...     ((1.0, -1.0), (0.0, 0.0), (0.0, 0.0)),
    ...     closest_polarity=False,
    ... )[0, 1]  # doctest: +ELLIPSIS
    3.1415...
    >>> compute_pairwise_angles(
    ...     ((1.0, -1.0), (0.0, 0.0), (0.0, 0.0)),
    ... )[0, 1]
    0.0

    References
    ----------
    .. [Andersson15] J. L. R. Andersson. et al., An integrated approach to
       correction for off-resonance effects and subject movement in diffusion MR
       imaging, NeuroImage 125 (2016) 1063–1078

    """

    bvecs_X = gtab_X.bvecs.T
    bvecs_X = np.array(bvecs_X) / np.linalg.norm(bvecs_X, axis=0)

    if gtab_Y is None:
        bvecs_Y = bvecs_X
    else:
        bvecs_Y = gtab_Y.bvecs.T
        bvecs_Y = np.array(bvecs_Y) / np.linalg.norm(bvecs_Y, axis=0)

    cosines = np.clip(bvecs_X.T @ bvecs_Y, -1.0, 1.0)
    return np.arccos(np.abs(cosines) if closest_polarity else cosines)
