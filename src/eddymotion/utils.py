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

from enum import Enum

import numpy as np


class SortingStrategy(Enum):
    """
    Enum class representing different sorting strategies.

    Available sorting strategies:
    - LINEAR: Sorts the items in a linear order.
    - RANDOM: Sorts the items in a random order.
    - BVALUE: Sorts the items based on their b-value.
    - CENTRALSYM: Sorts the items based on their central symmetry.
    """

    LINEAR = "linear"
    RANDOM = "random"
    BVALUE = "bvalue"
    CENTRALSYM = "centralsym"



def linear_action(size, **kwargs):
    """
    Sort the DWI data volume indices linearly

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
    >>> linear_action(10)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    """
    index_order = np.arange(len(dwdata))

    return index_order


def random_action(dwdata, seed=None):
    """Sort the DWI data volume indices.

    Parameters
    ----------
    dwdata : :obj:`~eddymotion.dmri.DWI` DWI dataset, represented by this tool's internal type.
    seed : :obj:`int` or :obj:`bool`, optional
        Seed the random number generator. If an integer, the value is used to
        initialize the generator; if ``True``, the arbitrary value
        of ``20210324`` is used to initialize it.

    Returns
    -------
    index_order : :obj:`numpy.ndarray`
        The sorted index order.
    """

    _seed = kwargs.get('seed', None)
    if seed or seed == 0:
        _seed = 20210324 if seed is True else seed

    rng = np.random.default_rng(_seed)

    index_order = np.arange(len(dwdata))
    rng.shuffle(index_order)

    return index_order.to_list()


def bvalue_action(dwdata):
    """
    Sort the DWI data volume indices in ascending order based on the last column of gradients.

    Parameters
    ----------
    dwdata : :obj:`~eddymotion.dmri.DWI` DWI dataset, represented by this tool's internal type.

    Returns
    -------
    numpy.ndarray: The sorted index order.
    """
    last_column = dwdata.gradients[:, -1]
    index_order = np.argsort(last_column)
    return index_order


def centralsym_action(size, **kwargs):
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
    """
    old_index = np.arange(len(dwdata))

    index_order = old_index.copy()
    if len(old_index) % 2 == 0:
        middle_point = int(len(old_index) / 2 - 1)
        index_order[0] = old_index[middle_point]

        for i in np.arange(1, middle_point+1):
            index_order[2 * i - 1] = old_index[middle_point + i]
            index_order[2 * i] = old_index[middle_point - i]
    else:
        middle_point = int(len(old_index) / 2)
        index_order[0] = old_index[middle_point]
        for i in np.arange(1, middle_point + 1):
            index_order[2 * i - 1] = old_index[middle_point + i]
            index_order[2 * i] = old_index[middle_point - i]

    return index_order
