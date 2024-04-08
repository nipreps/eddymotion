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
"""Data splitting helpers."""

from pathlib import Path

import h5py
import numpy as np


def lovo_split(dataset, index, with_b0=False):
    """
    Produce one fold of LOVO (leave-one-volume-out).

    Parameters
    ----------
    dataset : :obj:`eddymotion.data.dmri.DWI`
        DWI object
    index : :obj:`int`
        Index of the DWI orientation to be left out in this fold.

    Returns
    -------
    (train_data, train_gradients) : :obj:`tuple`
        Training DWI and corresponding gradients.
        Training data/gradients come **from the updated dataset**.
    (test_data, test_gradients) :obj:`tuple`
        Test 3D map (one DWI orientation) and corresponding b-vector/value.
        The test data/gradient come **from the original dataset**.

    """

    if not Path(dataset.get_filename()).exists():
        dataset.to_filename(dataset.get_filename())

    # read original DWI data & b-vector
    with h5py.File(dataset.get_filename(), "r") as in_file:
        root = in_file["/0"]
        data = np.asanyarray(root["dataobj"])
        gradients = np.asanyarray(root["gradients"])

    # if the size of the mask does not match data, cache is stale
    mask = np.zeros(data.shape[-1], dtype=bool)
    mask[index] = True

    train_data = data[..., ~mask]
    train_gradients = gradients[..., ~mask]
    test_data = data[..., mask]
    test_gradients = gradients[..., mask]

    if with_b0:
        train_data = np.concatenate(
            (np.asanyarray(dataset.bzero)[..., np.newaxis], train_data),
            axis=-1,
        )
        b0vec = np.zeros((4, 1))
        b0vec[0, 0] = 1
        train_gradients = np.concatenate(
            (b0vec, train_gradients),
            axis=-1,
        )

    return (
        (train_data, train_gradients),
        (test_data, test_gradients),
    )
