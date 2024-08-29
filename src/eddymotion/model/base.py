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
"""Base infrastructure for eddymotion's models."""

import numpy as np

from eddymotion.exceptions import ModelNotFittedError


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
            return TrivialModel(predicted=kwargs.pop("S0"), gtab=kwargs.pop("gtab"))

        if model.lower() in ("avgdwi", "averagedwi", "meandwi"):
            from eddymotion.model.dmri import AverageDWModel

            return AverageDWModel(**kwargs)

        if model.lower() in ("avg", "average", "mean"):
            return AverageModel(**kwargs)

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


class TrivialModel(BaseModel):
    """A trivial model that returns a given map always."""

    __slots__ = ("_predicted",)

    def __init__(self, predicted=None, **kwargs):
        """Implement object initialization."""
        if predicted is None:
            raise TypeError("This model requires the predicted map at initialization")

        super().__init__(**kwargs)
        self._predicted = predicted
        self._datashape = predicted.shape

    @property
    def is_fitted(self):
        return True

    def fit(self, data, **kwargs):
        """Do nothing."""

    def predict(self, *_, **kwargs):
        """Return the *b=0* map."""

        # No need to check fit (if not fitted, has raised already)
        return self._predicted


class AverageModel(BaseModel):
    """A trivial model that returns an average map."""

    __slots__ = ("_data",)

    def __init__(self, **kwargs):
        """Initialize a new model."""
        super().__init__(**kwargs)
        self._data = None

    def fit(self, data, **kwargs):
        """Calculate the average."""

        # Regress out global signal differences
        if kwargs.pop("equalize", False):
            data = data.copy().astype("float32")
            reshaped_data = (
                data.reshape((-1, data.shape[-1])) if self._mask is None else data[self._mask]
            )
            p5 = np.percentile(reshaped_data, 5.0, axis=0)
            p95 = np.percentile(reshaped_data, 95.0, axis=0) - p5
            data = (data - p5) * p95.mean() / p95 + p5.mean()

        # Select the summary statistic
        avg_func = getattr(np, kwargs.pop("stat", "mean"))

        # Calculate the average
        self._data = avg_func(data, axis=-1)

    @property
    def is_fitted(self):
        return self._data is not None

    def predict(self, *_, **kwargs):
        """Return the average map."""

        if self._data is None:
            raise ModelNotFittedError(f"{type(self).__name__} must be fitted before predicting")

        return self._data
