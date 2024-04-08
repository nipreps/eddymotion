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
"""Utils to sort the DWI data volume indices"""

import numpy as np
from itertools import chain, zip_longest


def linear_iterator(size=None, **kwargs):
    """
    Sort the DWI data volume indices linearly

    Parameters
    ----------
    size : :obj:`int`
        Number of volumes in the dataset
        (for instance, the number of orientations in a DWI)

    Returns
    -------
    :obj:`~typing.Generator`
        The sorted index order.

    Examples
    --------
    >>> list(linear_iterator(10))
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    """
    if size is None and 'bvals' in kwargs:
        size = len(kwargs['bvals'])
    if size is None:
        raise TypeError("Cannot build iterator without size")

    return range(size)


def random_iterator(size=None, **kwargs):
    """Sort the DWI data volume indices.

    Parameters
    ----------
    size : :obj:`int`
        Number of volumes in the dataset
        (for instance, the number of orientations in a DWI)
    test_seed : :obj:`bool`
        If True, return the seed value used for the random number generator.

    Returns
    -------
    :obj:`list` of :obj:`int`
        The sorted index order.

    Examples
    --------
    >>> random_iterator(10, seed=0, test_seed=True)
    20210324
    >>> random_iterator(10, seed=True, test_seed=True)
    20210324
    >>> random_iterator(10, seed=42)
    42

    """

    if size is None and 'bvals' in kwargs:
        size = len(kwargs['bvals'])
    if size is None:
        raise TypeError("Cannot build iterator without size")

    _seed = kwargs.get('seed', None)
    if _seed is True or _seed == 0:
        _seed = 20210324 if kwargs.get('seed', None) in [True, 0] else kwargs.get('seed', None)

    rng = np.random.default_rng(_seed)

    index_order = np.arange(size)
    rng.shuffle(index_order)

    if kwargs.get('test_seed', False) is True:
        return _seed

    return index_order.to_list()


def bvalue_iterator(size=None, **kwargs):
    """
    Sort the DWI data volume indices in ascending order based on the last column of gradients.

    Parameters
    ----------
    bvalues : :obj:`list`
        List of b-values corresponding to all orientations of the dataset.

    Returns
    -------
    :obj:`list` of :obj:`int`
        The sorted index order.

    Examples
    --------
    >>> list(bvalue_iterator(bvals=[0.0, 0.0, 1000.0, 1000.0, 700.0, 700.0, 2000.0, 2000.0, 0.0]))
    [0, 1, 8, 4, 5, 2, 3, 6, 7]

    """
    bvals = kwargs.get('bvals', None)
    if bvals is None:
        raise TypeError('Keyword argument bvals is required')
    indexed_bvals = sorted([(round(b, 2), i) for i, b in enumerate(bvals)])
    return (index[1] for index in indexed_bvals)


def centralsym_iterator(size=None, **kwargs):
    """
    Sort the DWI data volume indices in a central symmetric manner.

    Parameters
    ----------
    size : :obj:`int`
        Number of volumes in the dataset
        (for instance, the number of orientations in a DWI)

    Returns
    -------
    :obj:`list` of :obj:`int`
        The sorted index order.

    Examples
    --------
    >>> list(centralsym_iterator(10))
    [5, 4, 6, 3, 7, 2, 8, 1, 9, 0]
    >>> list(centralsym_iterator(11))
    [5, 4, 6, 3, 7, 2, 8, 1, 9, 0, 10]

    """
    if size is None and 'bvals' in kwargs:
        size = len(kwargs['bvals'])
    if size is None:
        raise TypeError("Cannot build iterator without size")
    linear = list(range(size))
    return (
        x for x in chain.from_iterable(zip_longest(
            linear[size // 2:],
            reversed(linear[:size // 2]),
        ))
        if x is not None
    )
