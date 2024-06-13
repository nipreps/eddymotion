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

DEFAULT_NUM_BINS = 15
"""Number of bins to classify b-values."""

DEFAULT_MULTISHELL_BIN_COUNT_THR = 7
"""Default bin count to consider a multishell scheme."""

DEFAULT_MAX_BVAL = 8000
"""Maximum b-value cap."""


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
