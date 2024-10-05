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
from joblib import Parallel, delayed

from eddymotion.exceptions import ModelNotFittedError
from eddymotion.model._dipy import _rasb2dipy
from eddymotion.model.base import BaseModel


def _exec_fit(model, data, chunk=None):
    retval = model.fit(data)
    return retval, chunk


def _exec_predict(model, chunk=None, **kwargs):
    """Propagate model parameters and call predict."""
    return np.squeeze(model.predict(**kwargs)), chunk


DEFAULT_CLIP_PERCENTILE = 75
"""Upper percentile threshold for intensity clipping."""

DEFAULT_MIN_S0 = 1e-5
"""Minimum value when considering the :math:`S_{0}` DWI signal."""

DEFAULT_MAX_S0 = 1.0
"""Maximum value when considering the :math:`S_{0}` DWI signal."""

DEFAULT_MAX_BVALUE = 1000
"""Maximum allowed value for the b-value."""

DEFAULT_LOWB_THRESHOLD = 50
"""The lower bound for the b-value so that the orientation is considered a DW volume."""

DEFAULT_HIGHB_THRESHOLD = 10000
"""A b-value cap for DWI data."""

DEFAULT_NUM_BINS = 15
"""Number of bins to classify b-values."""

DEFAULT_MULTISHELL_BIN_COUNT_THR = 7
"""Default bin count to consider a multishell scheme."""

DEFAULT_MAX_BVAL = 8000
"""Maximum b-value cap."""


class BaseDWIModel(BaseModel):
    """Interface and default methods for DWI models."""

    __slots__ = (
        "_gtab",
        "_S0",
        "_b_max",
        "_model_class",  # Defining a model class, DIPY models are instantiated automagically
        "_modelargs",
    )

    def __init__(self, gtab, S0=None, b_max=None, **kwargs):
        """Initialization.

        Parameters
        ----------
        gtab : :obj:`numpy.ndarray`
            An :math:`N \times 4` table, where rows (*N*) are diffusion gradients and
            columns are b-vector components and corresponding b-value, respectively.
        S0 : :obj:`numpy.ndarray`
            :math:`S_{0}` signal.
        b_max : :obj:`int`
            Maximum value to cap b-values.

        """

        super().__init__(**kwargs)

        # Setup B0 map
        self._S0 = None
        if S0 is not None:
            self._S0 = np.clip(
                S0.astype("float32") / S0.max(),
                a_min=DEFAULT_MIN_S0,
                a_max=DEFAULT_MAX_S0,
            )

        # Cap b-values, if requested
        self._gtab = gtab
        self._b_max = None
        if b_max and b_max > DEFAULT_MAX_BVALUE:
            # Saturate b-values at b_max, since signal stops dropping
            self._gtab[-1, self._gtab[-1] > b_max] = b_max
            # A possibly good alternative is completely remove very high b-values
            # bval_mask = gtab[-1] < b_max
            # data = data[..., bval_mask]
            # gtab = gtab[:, bval_mask]
            self._b_max = b_max

        kwargs = {k: v for k, v in kwargs.items() if k in self._modelargs}

        # DIPY models (or one with a fully-compliant interface)
        model_str = getattr(self, "_model_class", None)
        if model_str:
            from importlib import import_module

            module_name, class_name = model_str.rsplit(".", 1)
            self._model = getattr(
                import_module(module_name),
                class_name,
            )(_rasb2dipy(gtab), **kwargs)

    def fit(self, data, n_jobs=None, **kwargs):
        """Fit the model chunk-by-chunk asynchronously"""
        n_jobs = n_jobs or 1

        self._datashape = data.shape

        # Select voxels within mask or just unravel 3D if no mask
        data = (
            data[self._mask, ...] if self._mask is not None else data.reshape(-1, data.shape[-1])
        )

        # One single CPU - linear execution (full model)
        if n_jobs == 1:
            self._model, _ = _exec_fit(self._model, data)
            return

        # Split data into chunks of group of slices
        data_chunks = np.array_split(data, n_jobs)

        self._models = [None] * n_jobs

        # Parallelize process with joblib
        with Parallel(n_jobs=n_jobs) as executor:
            results = executor(
                delayed(_exec_fit)(self._model, dchunk, i) for i, dchunk in enumerate(data_chunks)
            )
        for submodel, index in results:
            self._models[index] = submodel

        self._is_fitted = True
        self._model = None  # Preempt further actions on the model

    def predict(self, gradient=None, **kwargs):
        """Predict asynchronously chunk-by-chunk the diffusion signal."""

        if gradient is None:
            raise ValueError("A gradient to be simulated (b-vector, b-value) must be provided")

        if not self._is_fitted:
            raise ModelNotFittedError(f"{type(self).__name__} must be fitted before predicting")

        gradient = np.array(gradient)  # Tuples are unmutable

        # Cap the b-value if b_max is defined
        gradient[-1] = min(gradient[-1], self._b_max or gradient[-1])

        gradient = _rasb2dipy(gradient)

        S0 = None
        if self._S0 is not None:
            S0 = (
                self._S0[self._mask, ...]
                if self._mask is not None
                else self._S0.reshape(-1, self._S0.shape[-1])
            )

        n_models = len(self._models) if self._model is None and self._models else 1

        if n_models == 1:
            predicted, _ = _exec_predict(self._model, **(kwargs | {"gtab": gradient, "S0": S0}))
        else:
            S0 = np.array_split(S0, n_models) if S0 is not None else [None] * n_models

            predicted = [None] * n_models

            # Parallelize process with joblib
            with Parallel(n_jobs=n_models) as executor:
                results = executor(
                    delayed(_exec_predict)(
                        model,
                        chunk=i,
                        **(kwargs | {"gtab": gradient, "S0": S0[i]}),
                    )
                    for i, model in enumerate(self._models)
                )
            for subprediction, index in results:
                predicted[index] = subprediction

            predicted = np.hstack(predicted)

        if self._mask is not None:
            retval = np.zeros_like(self._mask, dtype="float32")
            retval[self._mask, ...] = predicted
        else:
            retval = predicted.reshape(self._datashape[:-1])

        return retval


class AverageDWModel(BaseDWIModel):
    """A trivial model that returns an average map."""

    __slots__ = ("_data", "_th_low", "_th_high", "_bias", "_stat", "_is_fitted")

    def __init__(self, **kwargs):
        r"""
        Implement object initialization.

        Parameters
        ----------
        th_low : :obj:`numbers.Number`
            A lower bound for the b-value corresponding to the diffusion weighted images
            that will be averaged.
        th_high : :obj:`numbers.Number`
            An upper bound for the b-value corresponding to the diffusion weighted images
            that will be averaged.
        bias : :obj:`bool`
            Whether the overall distribution of each diffusion weighted image will be
            standardized and centered around the
            :data:`src.eddymotion.model.base.DEFAULT_CLIP_PERCENTILE` percentile.
        stat : :obj:`str`
            Whether the summary statistic to apply is ``"mean"`` or ``"median"``.

        """
        super().__init__(**kwargs)

        self._th_low = kwargs.get("th_low", DEFAULT_LOWB_THRESHOLD)
        self._th_high = kwargs.get("th_high", DEFAULT_HIGHB_THRESHOLD)
        self._bias = kwargs.get("bias", True)
        self._stat = kwargs.get("stat", "median")
        self._data = None

    def fit(self, data, **kwargs):
        """Calculate the average."""

        if (gtab := kwargs.pop("gtab", None)) is None:
            raise ValueError("A gradient table must be provided.")

        # Select the interval of b-values for which DWIs will be averaged
        b_mask = (
            ((gtab[3] >= self._th_low) & (gtab[3] <= self._th_high))
            if gtab is not None
            else np.ones((data.shape[-1],), dtype=bool)
        )
        shells = data[..., b_mask]

        # Regress out global signal differences
        if self._bias:
            centers = np.median(shells, axis=(0, 1, 2))
            reference = np.percentile(centers[centers >= 1.0], DEFAULT_CLIP_PERCENTILE)
            centers[centers < 1.0] = reference
            drift = reference / centers
            shells = shells * drift

        # Select the summary statistic
        avg_func = np.median if self._stat == "median" else np.mean
        # Calculate the average
        self._data = avg_func(shells, axis=-1)
        self._is_fitted = True

    def predict(self, *_, **kwargs):
        """Return the average map."""

        if not self._is_fitted:
            raise ModelNotFittedError(f"{type(self).__name__} must be fitted before predicting")

        return self._data


class DTIModel(BaseDWIModel):
    """A wrapper of :obj:`dipy.reconst.dti.TensorModel`."""

    _modelargs = (
        "min_signal",
        "return_S0_hat",
        "fit_method",
        "weighting",
        "sigma",
        "jac",
    )
    _model_class = "dipy.reconst.dti.TensorModel"


class DKIModel(BaseDWIModel):
    """A wrapper of :obj:`dipy.reconst.dki.DiffusionKurtosisModel`."""

    _modelargs = DTIModel._modelargs
    _model_class = "dipy.reconst.dki.DiffusionKurtosisModel"


class GPModel(BaseDWIModel):
    """A wrapper of :obj:`~eddymotion.model.dipy.GaussianProcessModel`."""

    _modelargs = ("kernel_model",)
    _model_class = "eddymotion.model._dipy.GaussianProcessModel"


def find_shelling_scheme(
    bvals,
    num_bins=DEFAULT_NUM_BINS,
    multishell_nonempty_bin_count_thr=DEFAULT_MULTISHELL_BIN_COUNT_THR,
    bval_cap=DEFAULT_MAX_BVAL,
):
    """
    Find the shelling scheme on the given b-values.

    Computes the histogram of the b-values according to ``num_bins``
    and depending on the nonempty bin count, classify the shelling scheme
    as single-shell if they are 2 (low-b and a shell); multi-shell if they are
    below the ``multishell_nonempty_bin_count_thr`` value; and DSI otherwise.

    Parameters
    ----------
    bvals : :obj:`list` or :obj:`~numpy.ndarray`
         List or array of b-values.
    num_bins : :obj:`int`, optional
        Number of bins.
    multishell_nonempty_bin_count_thr : :obj:`int`, optional
        Bin count to consider a multi-shell scheme.

    Returns
    -------
    scheme : :obj:`str`
        Shelling scheme.
    bval_groups : :obj:`list`
        List of grouped b-values.
    bval_estimated : :obj:`list`
        List of 'estimated' b-values as the median value of each b-value group.

    """

    # Bin the b-values: use -1 as the lower bound to be able to appropriately
    # include b0 values
    hist, bin_edges = np.histogram(bvals, bins=num_bins, range=(-1, min(max(bvals), bval_cap)))

    # Collect values in each bin
    bval_groups = []
    bval_estimated = []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:], strict=False):
        # Add only if a nonempty b-values mask
        if (mask := (bvals > lower) & (bvals <= upper)).sum():
            bval_groups.append(bvals[mask])
            bval_estimated.append(np.median(bvals[mask]))

    nonempty_bins = len(bval_groups)

    if nonempty_bins < 2:
        raise ValueError("DWI must have at least one high-b shell")

    if nonempty_bins == 2:
        scheme = "single-shell"
    elif nonempty_bins < multishell_nonempty_bin_count_thr:
        scheme = "multi-shell"
    else:
        scheme = "DSI"

    return scheme, bval_groups, bval_estimated
