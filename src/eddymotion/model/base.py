# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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
"""A factory class that adapts DIPY's dMRI models."""

import warnings

import numpy as np
from dipy.core.gradients import gradient_table
from joblib import Parallel, delayed

from eddymotion.exceptions import ModelNotFittedError

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

DEFAULT_CLIP_PERCENTILE = 75
"""Upper percentile threshold for intensity clipping."""

DEFAULT_TIMEFRAME_MIDPOINT_TOL = 1e-2
"""Time frame tolerance in seconds."""


def _exec_fit(model, data, chunk=None):
    retval = model.fit(data)
    return retval, chunk


def _exec_predict(model, chunk=None, **kwargs):
    """Propagate model parameters and call predict."""
    return np.squeeze(model.predict(**kwargs)), chunk


class ModelFactory:
    """A factory for instantiating diffusion models."""

    @staticmethod
    def init(model="DTI", **kwargs):
        """
        Instantiate a diffusion model.

        Parameters
        ----------
        model : :obj:`str`
            Diffusion model.
            Options: ``"DTI"``, ``"DKI"``, ``"S0"``, ``"AverageDW"``

        Return
        ------
        model : :obj:`~dipy.reconst.ReconstModel`
            A model object compliant with DIPY's interface.

        """
        if model.lower() in ("s0", "b0"):
            return TrivialB0Model(S0=kwargs.pop("S0"), gtab=kwargs.pop("gtab"))

        if model.lower() in ("avg", "average", "mean"):
            return AverageDWModel(**kwargs)

        if model.lower() in ("dti", "dki", "pet"):
            Model = globals()[f"{model.upper()}Model"]
            return Model(**kwargs)

        raise NotImplementedError(f"Unsupported model <{model}>.")


class BaseModel:
    """
    Defines the interface and default methods.

    Implements the interface of :obj:`dipy.reconst.base.ReconstModel`.
    Instead of inheriting from the abstract base, this implementation
    follows type adaptation principles, as it is easier to maintain
    and to read (see https://www.youtube.com/watch?v=3MNVP9-hglc).

    """

    __slots__ = (
        "_model",
        "_mask",
        "_models",
        "_datashape",
        "_is_fitted",
        "_modelargs",
    )

    def __init__(self, mask=None, **kwargs):
        """Base initialization."""

        # Keep model state
        self._model = None  # "Main" model
        self._models = None  # For parallel (chunked) execution

        # Setup brain mask
        self._mask = mask

        self._datashape = None
        self._is_fitted = False

        self._modelargs = ()

    @property
    def is_fitted(self):
        return self._is_fitted

    def fit(self, data, **kwargs):
        """Abstract member signature of fit()."""
        raise NotImplementedError("Cannot call fit() on a BaseModel instance.")

    def predict(self, *args, **kwargs):
        """Abstract member signature of predict()."""
        raise NotImplementedError("Cannot call predict() on a BaseModel instance.")


class PETModel(BaseModel):
    """A PET imaging realignment model based on B-Spline approximation."""

    __slots__ = ("_t", "_x", "_xlim", "_order", "_coeff", "_n_ctrl")

    def __init__(self, timepoints=None, xlim=None, n_ctrl=None, order=3, **kwargs):
        """
        Create the B-Spline interpolating matrix.

        Parameters:
        -----------
        timepoints : :obj:`list`
            The timing (in sec) of each PET volume.
            E.g., ``[15.,   45.,   75.,  105.,  135.,  165.,  210.,  270.,  330.,
            420.,  540.,  750., 1050., 1350., 1650., 1950., 2250., 2550.]``

        n_ctrl : :obj:`int`
            Number of B-Spline control points. If `None`, then one control point every
            six timepoints will be used. The less control points, the smoother is the
            model.

        """
        super.__init__(**kwargs)

        if timepoints is None or xlim is None:
            raise TypeError("timepoints must be provided in initialization")

        self._order = order

        self._x = np.array(timepoints, dtype="float32")
        self._xlim = xlim

        if self._x[0] < DEFAULT_TIMEFRAME_MIDPOINT_TOL:
            raise ValueError("First frame midpoint should not be zero or negative")
        if self._x[-1] > (self._xlim - DEFAULT_TIMEFRAME_MIDPOINT_TOL):
            raise ValueError("Last frame midpoint should not be equal or greater than duration")

        # Calculate index coordinates in the B-Spline grid
        self._n_ctrl = n_ctrl or (len(timepoints) // 4) + 1

        # B-Spline knots
        self._t = np.arange(-3, float(self._n_ctrl) + 4, dtype="float32")

        self._coeff = None

    @property
    def is_fitted(self):
        return self._coeff is not None

    def fit(self, data, **kwargs):
        """Fit the model."""
        from scipy.interpolate import BSpline
        from scipy.sparse.linalg import cg

        n_jobs = kwargs.pop("n_jobs", None) or 1

        timepoints = kwargs.get("timepoints", None) or self._x
        x = (np.array(timepoints, dtype="float32") / self._xlim) * self._n_ctrl

        self._datashape = data.shape[:3]

        # Convert data into V (voxels) x T (timepoints)
        data = data.reshape((-1, data.shape[-1])) if self._mask is None else data[self._mask]

        # A.shape = (T, K - 4); T= n. timepoints, K= n. knots (with padding)
        A = BSpline.design_matrix(x, self._t, k=self._order)
        AT = A.T
        ATdotA = AT @ A

        # One single CPU - linear execution (full model)
        if n_jobs == 1:
            self._coeff = np.array([cg(ATdotA, AT @ v)[0] for v in data])
            return

        # Parallelize process with joblib
        with Parallel(n_jobs=n_jobs) as executor:
            results = executor(delayed(cg)(ATdotA, AT @ v) for v in data)

        self._coeff = np.array([r[0] for r in results])

    def predict(self, index=None, **kwargs):
        """Return the corrected volume using B-spline interpolation."""
        from scipy.interpolate import BSpline

        if index is None:
            raise ValueError("A timepoint index to be simulated must be provided.")

        if not self._is_fitted:
            raise ModelNotFittedError(f"{type(self).__name__} must be fitted before predicting")

        # Project sample timing into B-Spline coordinates
        x = (index / self._xlim) * self._n_ctrl
        A = BSpline.design_matrix(x, self._t, k=self._order)

        # A is 1 (num. timepoints) x C (num. coeff)
        # self._coeff is V (num. voxels) x K - 4
        predicted = np.squeeze(A @ self._coeff.T)

        if self._mask is None:
            return predicted.reshape(self._datashape)

        retval = np.zeros(self._datashape, dtype="float32")
        retval[self._mask] = predicted
        return retval


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


class TrivialB0Model(BaseDWIModel):
    """A trivial model that returns a *b=0* map always."""

    def __init__(self, **kwargs):
        """Implement object initialization."""
        super().__init__(**kwargs)

        if self._S0 is None:
            raise ValueError("S0 must be provided")

    @property
    def is_fitted(self):
        return True

    def fit(self, data, **kwargs):
        """Do nothing."""

    def predict(self, *_, **kwargs):
        """Return the *b=0* map."""

        # No need to check fit (if not fitted, has raised already)
        return self._S0


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
    _model_class = "eddymotion.model.dipy.GaussianProcessModel"


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
        warnings.filterwarnings("ignore", category=UserWarning)
        retval = gradient_table(gradient[3, :], gradient[:3, :].T)
    return retval
