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
"""DIPY-like models (a sandbox to trial them out before upstreaming to DIPY)."""

from __future__ import annotations

import warnings

import numpy as np
from dipy.core.gradients import GradientTable
from dipy.reconst.base import ReconstModel
from sklearn.gaussian_process import GaussianProcessRegressor

from nifreeze.model.gpr import (
    EddyMotionGPR,
    ExponentialKriging,
    SphericalKriging,
)


def gp_prediction(
    model: GaussianProcessRegressor,
    gtab: GradientTable | np.ndarray,
    mask: np.ndarray | None = None,
    return_std: bool = False,
) -> np.ndarray:
    """
    Predicts one or more DWI orientations given a model.

    This function checks if the model is fitted and then extracts
    orientations and potentially b-values from the X. It predicts the mean
    and standard deviation of the DWI signal using the model.

    Parameters
    ----------
    model : :obj:`~sklearn.gaussian_process.GaussianProcessRegressor`
        A fitted GaussianProcessRegressor model.
    gtab : :obj:`~dipy.core.gradients.GradientTable` or :obj:`~np.ndarray`
        Gradient table with one or more orientations at which the GP will be evaluated.
    mask : :obj:`numpy.ndarray`, optional
        A boolean mask indicating which voxels to use (optional).
    return_std : bool, optional
        Whether to return the standard deviation of the predicted signal.

    Returns
    -------
    :obj:`numpy.ndarray`
        A 3D or 4D array with the simulated gradient(s).

    """

    X = gtab.bvecs.T if hasattr(gtab, "bvecs") else np.asarray(gtab)

    # Check it's fitted as they do in sklearn internally
    # https://github.com/scikit-learn/scikit-learn/blob/972e17fe1aa12d481b120ad4a3dc076bae736931/\
    # sklearn/gaussian_process/_gpr.py#L410C9-L410C42
    if not hasattr(model, "X_train_"):
        raise RuntimeError("Model is not yet fitted.")

    # Extract orientations from bvecs, and highly likely, the b-value too.
    return model.predict(X, return_std=return_std)


class GaussianProcessModel(ReconstModel):
    """A Gaussian Process (GP) model to simulate single- and multi-shell DWI data."""

    __slots__ = (
        "kernel",
        "_modelfit",
    )

    def __init__(
        self,
        kernel_model: str = "spherical",
        beta_l: float = 2.0,
        beta_a: float = 0.1,
        sigma_sq: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        """A GP-based DWI model [Andersson15]_.

        Parameters
        ----------
        kernel_model : :obj:`~sklearn.gaussian_process.kernels.Kernel`, optional
            Kernel model to calculate the GP's covariance matrix.
        lambda_s : :obj:`float`, optional
            Signal scale parameter determining the variability of the signal.
        a : :obj:`float`, optional
            Distance scale parameter determining how fast the covariance
            decreases as one moves along the surface of the sphere. Must have a
            positive value.
        sigma_sq : :obj:`float`, optional
            Uncertainty of the measured values.

        References
        ----------
        .. [Andersson15] Jesper L.R. Andersson and Stamatios N. Sotiropoulos.
           Non-parametric representation and prediction of single- and multi-shell
           diffusion-weighted MRI data using Gaussian processes. NeuroImage, 122:166-176, 2015.
           doi:\
           `10.1016/j.neuroimage.2015.07.067 <https://doi.org/10.1016/j.neuroimage.2015.07.067>`__.

        """

        ReconstModel.__init__(self, None)

        self.sigma_sq = sigma_sq

        KernelType = SphericalKriging if kernel_model == "spherical" else ExponentialKriging
        self.kernel = KernelType(
            beta_a=beta_a,
            beta_l=beta_l,
        )

    def fit(
        self,
        data: np.ndarray,
        gtab: GradientTable | np.ndarray,
        mask: np.ndarray[bool] | None = None,
        random_state: int = 0,
    ) -> GPFit:
        """Fit method of the DTI model class

        Parameters
        ----------
        gtab : :obj:`~dipy.core.gradients.GradientTable` or :obj:`~np.ndarray`
            The gradient table corresponding to the training data.
        data : :obj:`~numpy.ndarray`
            The measured signal from one voxel.
        mask : :obj:`~numpy.ndarray`
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]
        random_state: :obj:`int`, optional
            Determines random number generation used to initialize the centers
            of the kernel bounds.

        Returns
        -------
        :obj:`~nifreeze.model.dipy.GPFit`
            A model fit container object.

        """

        # Extract b-vecs: scikit-learn wants (n_samples, n_features)
        # where n_features is 3, and n_samples the different diffusion-encoding
        # gradient orientations.
        X = gtab.bvecs if hasattr(gtab, "bvecs") else np.asarray(gtab)

        # Data must have shape (n_samples, n_targets) where n_samples is
        # the number of diffusion-encoding gradient orientations, and n_targets
        # is number of voxels.
        y = (
            data[mask[..., None]] if mask is not None else np.reshape(data, (-1, data.shape[-1]))
        ).T

        if (grad_dirs := X.shape[0]) != (signal_dirs := y.shape[0]):
            raise ValueError(
                f"Mismatched gradient directions in data ({signal_dirs}) "
                f"and gradient table ({grad_dirs})."
            )

        gpr = EddyMotionGPR(
            kernel=self.kernel,
            random_state=random_state,
            n_targets=y.shape[1],
            alpha=self.sigma_sq,
        )
        self._modelfit = GPFit(
            model=gpr.fit(X, y),
            mask=mask,
        )
        return self._modelfit

    def predict(
        self,
        gtab: GradientTable | np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict using the Gaussian process model of the DWI signal for one or more gradients.

        Parameters
        ----------
        gtab : :obj:`~dipy.core.gradients.GradientTable` or :obj:`~np.ndarray`
            Gradient table with one or more orientations at which the GP will be evaluated.

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
    model : :obj:`~sklearn.gaussian_process.GaussianProcessRegressor`
        The fitted Gaussian process regressor object.
    mask : :obj:`~numpy.ndarray`
        The boolean mask used during fitting (can be ``None``).

    """

    def __init__(
        self,
        model: GaussianProcessRegressor,
        mask: np.ndarray | None = None,
    ) -> None:
        """
        Initialize a Gaussian Process fit container.

        Parameters
        ----------
        model : :obj:`~sklearn.gaussian_process.GaussianProcessRegressor`
            The fitted Gaussian process regressor object.
        mask : :obj:`~numpy.ndarray`, optional
            The boolean mask used during fitting.

        """
        self.model = model
        self.mask = mask

    def predict(
        self,
        gtab: GradientTable | np.ndarray,
    ) -> np.ndarray:
        """
        Generate DWI signal based on a fitted Gaussian Process.

        Parameters
        ----------
        gtab : :obj:`~dipy.core.gradients.GradientTable` or :obj:`~np.ndarray`
            Gradient table with one or more orientations at which the GP will be evaluated.

        Returns
        -------
        :obj:`numpy.ndarray`
            A 3D or 4D array with the simulated gradient(s).

        """
        return gp_prediction(self.model, gtab, mask=self.mask)


def _rasb2dipy(gradient):
    gradient = np.asanyarray(gradient)
    if gradient.ndim == 1:
        if gradient.size != 4:
            raise ValueError("Missing gradient information.")
        gradient = gradient[..., np.newaxis]

    if gradient.shape[0] != 4:
        gradient = gradient.T
    elif gradient.shape == (4, 4):
        print("Warning: make sure gradient information is not transposed!")

    with warnings.catch_warnings():
        from dipy.core.gradients import gradient_table

        warnings.filterwarnings("ignore", category=UserWarning)
        retval = gradient_table(gradient[3, :], gradient[:3, :].T)
    return retval
