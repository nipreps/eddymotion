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

from inspect import isclass

from eddymotion.exceptions import NotFittedError

not_fitted_msg = (
    "This %(name)s instance is not fitted yet. Call 'fit' with appropriate "
    "arguments before using this model."
)
not_instance = "%(name)s is a class, not an instance."
not_model_instance = "%(name)s is not a model instance."


def _is_fitted(model, attributes=None, all_or_any=all):
    """Determine if a model is fitted.

    Parameters
    ----------
    model : model instance
        Model instance for which the check is performed.
    attributes : obj:`str`, obj:`list` or obj:`tuple`
        Attribute name(s) given as string or a list/tuple of strings
        e.g.: ``["coef_", "model_", ...], "coef_"``.
        If `None`, `model` is considered fitted if the instance has set its
        state to being fitted.
    all_or_any : callable, {obj:`all`, obj:`any`}
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    obj:`bool`
        ``True`` if the model is fitted; ``False`` otherwise.
    """

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        return all_or_any([hasattr(model, attr) for attr in attributes])

    return model.__model_is_fitted__()


def check_is_fitted(model, attributes=None, *, msg=None, all_or_any=all):
    """Perform is_fitted validation for model.

    Checks if the model is fitted by verifying the presence of fitted attributes
    (ending with a trailing underscore) and otherwise raises a NotFittedError
    with the given message.

    If a model does not set any attributes with a trailing underscore, it can
    define a ``__model_is_fitted__`` method returning a boolean to specify if
    the model is fitted or not.

    Parameters
    ----------
    model : model instance
        Model instance for which the check is performed.
    attributes : obj:`str`, obj:`list` or obj:`tuple`
        Attribute name(s) given as string or a list/tuple of strings.
        e.g.: ``["coef_", "model_", ...], "coef_"``
        If `None`, `model` is considered fitted if the instance has set its
        state to being fitted.
    msg : obj:`str`
        The error message to be shown. A default message is shown if `None`. For
        custom messages if "%(name)s" is present in the message string, it is
        substituted for the model name.
        e.g.: "model %(name)s must be fitted before sparsifying".
    all_or_any : callable, {obj:`all`, obj:`any`}
        Specify whether all or any of the given attributes must exist.

    Raises
    ------
    TypeError
        If the model is a class or not a model instance.
    NotFittedError
        If the model has not been fitted.
    """

    if isclass(model):
        raise TypeError(not_instance % {"name": model})
    if msg is None:
        msg = not_fitted_msg

    if not hasattr(model, "fit"):
        raise TypeError(not_model_instance % {"name": model})

    if not _is_fitted(model, attributes, all_or_any):
        raise NotFittedError(msg % {"name": type(model).__name__})
