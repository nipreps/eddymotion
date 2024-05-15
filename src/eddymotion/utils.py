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
"""Iterators to traverse the volumes in a 4D image."""

import random
from itertools import chain, zip_longest
from typing import Iterator


def linear_iterator(size: int = None, **kwargs) -> Iterator[int]:
    """
    Traverse the dataset volumes in ascending order.

    Parameters
    ----------
    size : :obj:`int`
        Number of volumes in the dataset
        (for instance, the number of orientations in a DWI)

    Returns
    -------
    :obj:`~typing.Iterator`
        The sorted index order.

    Examples
    --------
    >>> list(linear_iterator(10))
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    """
    if size is None and "bvals" in kwargs:
        size = len(kwargs["bvals"])
    if size is None:
        raise TypeError("Cannot build iterator without size")

    return range(size)


def random_iterator(size: int = None, **kwargs) -> Iterator[int]:
    """
    Traverse the dataset volumes randomly.

    Parameters
    ----------
    size : :obj:`int`
        Number of volumes in the dataset
        (for instance, the number of orientations in a DWI)
    seed : :obj:`int` or :obj:`bool` or :obj:`bool` or ``None``
        If :obj:`int` or :obj:`str` or ``None``, initializes the seed of Python's random generator
        with the given value.
        If ``False``, the random generator is passed ``None``.
        If ``True``, a default seed value is set.

    Returns
    -------
    :obj:`~typing.Iterator`
        The sorted index order.

    Examples
    --------
    >>> list(random_iterator(15, seed=0))  # seed is 0
    [1, 10, 9, 5, 11, 2, 3, 7, 8, 4, 0, 14, 12, 6, 13]
    >>>  # seed is True -> the default value 20210324 is set
    >>> list(random_iterator(15, seed=True))
    [1, 12, 14, 5, 0, 11, 10, 9, 7, 8, 3, 13, 2, 6, 4]
    >>> list(random_iterator(15, seed=20210324))
    [1, 12, 14, 5, 0, 11, 10, 9, 7, 8, 3, 13, 2, 6, 4]
    >>> list(random_iterator(15, seed=42))  # seed is 42
    [8, 13, 7, 6, 14, 12, 5, 2, 9, 3, 4, 11, 0, 1, 10]

    """

    if size is None and "bvals" in kwargs:
        size = len(kwargs["bvals"])
    if size is None:
        raise TypeError("Cannot build iterator without size")

    _seed = kwargs.get("seed", None)
    _seed = 20210324 if _seed is True else _seed

    random.seed(None if _seed is False else _seed)

    index_order = list(range(size))
    random.shuffle(index_order)
    return (x for x in index_order)


def bvalue_iterator(size: int = None, **kwargs) -> Iterator[int]:
    """
    Traverse the volumes in a DWI dataset by growing b-value.

    Parameters
    ----------
    bvalues : :obj:`list`
        List of b-values corresponding to all orientations of the dataset.

    Returns
    -------
    :obj:`~typing.Iterator`
        The sorted index order.

    Examples
    --------
    >>> list(bvalue_iterator(bvals=[0.0, 0.0, 1000.0, 1000.0, 700.0, 700.0, 2000.0, 2000.0, 0.0]))
    [0, 1, 8, 4, 5, 2, 3, 6, 7]

    """
    bvals = kwargs.get("bvals", None)
    if bvals is None:
        raise TypeError("Keyword argument bvals is required")
    indexed_bvals = sorted([(round(b, 2), i) for i, b in enumerate(bvals)])
    return (index[1] for index in indexed_bvals)


def centralsym_iterator(size: int = None, **kwargs) -> Iterator[int]:
    """
    Traverse the dataset starting from the center and alternatingly progressing to the sides.

    Parameters
    ----------
    size : :obj:`int`
        Number of volumes in the dataset
        (for instance, the number of orientations in a DWI)

    Returns
    -------
    :obj:`~typing.Iterator`
        The sorted index order.

    Examples
    --------
    >>> list(centralsym_iterator(10))
    [5, 4, 6, 3, 7, 2, 8, 1, 9, 0]
    >>> list(centralsym_iterator(11))
    [5, 4, 6, 3, 7, 2, 8, 1, 9, 0, 10]

    """
    if size is None and "bvals" in kwargs:
        size = len(kwargs["bvals"])
    if size is None:
        raise TypeError("Cannot build iterator without size")
    linear = list(range(size))
    return (
        x
        for x in chain.from_iterable(
            zip_longest(
                linear[size // 2 :],
                reversed(linear[: size // 2]),
            )
        )
        if x is not None
    )
