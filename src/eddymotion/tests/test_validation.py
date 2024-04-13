# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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

import pytest

from eddymotion.exceptions import NotFittedError
from eddymotion.validation import check_is_fitted, not_fitted_msg, not_instance, not_model_instance


def test_not_instance():
    class MyClass:
        pass

    with pytest.raises(TypeError) as exc_info:
        check_is_fitted(MyClass)
    assert str(exc_info.value) == not_instance % {"name": MyClass}


def test_not_model_instance():
    class MyClass:
        pass

    my_class = MyClass()

    with pytest.raises(TypeError) as exc_info:
        check_is_fitted(my_class)
    assert str(exc_info.value) == not_model_instance % {"name": my_class}


def test_not_fitted_no_attr():
    class MyClassNoAttr:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            self._is_fitted = True

        def predict(self, *args, **kwargs):
            check_is_fitted(self)

        def __model_is_fitted__(self):
            return hasattr(self, "_is_fitted") and self._is_fitted

    my_class_no_attr = MyClassNoAttr()

    with pytest.raises(NotFittedError) as exc_info:
        my_class_no_attr.predict()
    assert str(exc_info.value) == not_fitted_msg % {"name": type(my_class_no_attr).__name__}

    my_class_no_attr.fit()
    my_class_no_attr.predict()


def test_not_fitted_attr_default():
    class MyClassAttrDef:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            self._coefs = [1, 2]

        def predict(self, *args, **kwargs):
            check_is_fitted(self, attributes="_coefs")

    my_class_attr_def = MyClassAttrDef()

    with pytest.raises(NotFittedError) as exc_info:
        my_class_attr_def.predict()
    assert str(exc_info.value) == not_fitted_msg % {"name": type(my_class_attr_def).__name__}

    my_class_attr_def.fit()
    my_class_attr_def.predict()


def test_not_fitted_attr_nondefault():
    class MyClassAttrAny:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            self._coefs = [1, 2]

        def predict(self, *args, **kwargs):
            check_is_fitted(self, attributes="_coefs", all_or_any=any)

    my_class_attr_any = MyClassAttrAny()

    with pytest.raises(NotFittedError) as exc_info:
        my_class_attr_any.predict()
    assert str(exc_info.value) == not_fitted_msg % {"name": type(my_class_attr_any).__name__}

    my_class_attr_any.fit()
    my_class_attr_any.predict()

    class MyClassAttrAll:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            self._coefs1 = [1, 2]

        def predict(self, *args, **kwargs):
            check_is_fitted(self, attributes=["_coefs1", "_coefs2"], all_or_any=all)

    my_class_attr_all = MyClassAttrAll()

    with pytest.raises(NotFittedError) as exc_info:
        my_class_attr_all.predict()
    assert str(exc_info.value) == not_fitted_msg % {"name": type(my_class_attr_all).__name__}

    my_class_attr_all.fit()

    with pytest.raises(NotFittedError) as exc_info:
        my_class_attr_all.predict()
    assert str(exc_info.value) == not_fitted_msg % {"name": type(my_class_attr_all).__name__}

    class MyClassAttrAll:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            self._coefs1 = [1, 2]
            self._coefs2 = [3, 4]

        def predict(self, *args, **kwargs):
            check_is_fitted(self, attributes=("_coefs1", "_coefs2"), all_or_any=all)

    my_class_attr_all = MyClassAttrAll()

    with pytest.raises(NotFittedError) as exc_info:
        my_class_attr_all.predict()
    assert str(exc_info.value) == not_fitted_msg % {"name": type(my_class_attr_all).__name__}

    my_class_attr_all.fit()
    my_class_attr_all.predict()


def test_not_fitted_msg():
    msg = "model %(name)s must be fitted before sparsifying"

    class MyClassMsg:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            self._coefs = [1, 2]

        def predict(self, *args, **kwargs):
            check_is_fitted(self, attributes="_coefs", msg=msg, all_or_any=any)

    my_class_msg = MyClassMsg()

    with pytest.raises(NotFittedError) as exc_info:
        my_class_msg.predict()
    assert str(exc_info.value) == msg % {"name": type(my_class_msg).__name__}

    my_class_msg.fit()
    my_class_msg.predict()
