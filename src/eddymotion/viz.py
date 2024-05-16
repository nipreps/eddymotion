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
"""Visualization utilities."""

import numpy as np
from nireports.reportlets.nuisance import plot_carpet as nw_plot_carpet


def plot_carpet(
    nii,
    gtab,
    segmentation=None,
    sort_by_bval=False,
    output_file=None,
    segment_labels=None,
    detrend=False,
):
    """Return carpet plot using niworkflows carpet_plot

    Parameters
    ----------
    nii : Nifti1Image
        DW imaging data
    gtab : :obj:`GradientTable`
        DW imaging data gradient data
    segmentation : Nifti1Image
        Boolean or segmentation mask of DW imaging data
    sort_by_bval : :obj:`bool`
        Flag to reorder time points by bvalue
    output_file : :obj:`string`
        Path to save the plot
    segment_labels : :obj:`dict`
        Dictionary of segment labels, mapping segment name to list of integers
        e.g. {'Cerebral_White_Matter': [2, 41], ...}
    detrend : :obj:`bool`
        niworkflows plot_carpet detrend flag

    Returns
    ---------
    matplotlib GridSpec object
    """
    segments = None

    nii_data = nii.get_fdata()

    b0_data = nii_data[..., gtab.b0s_mask]
    dw_data = nii_data[..., ~gtab.b0s_mask]

    bzero = np.mean(b0_data, -1)

    nii_data_div_b0 = dw_data / bzero[..., np.newaxis]

    sort_inds = np.argsort(
        gtab.bvals[~gtab.b0s_mask] if sort_by_bval else np.arange(len(gtab.bvals[~gtab.b0s_mask]))
    )
    nii_data_div_b0 = nii_data_div_b0[..., sort_inds]

    # Reshape
    nii_data_reshaped = nii_data_div_b0.reshape(-1, nii_data_div_b0.shape[-1])

    if segmentation is not None:
        segmentation_data = np.asanyarray(segmentation.dataobj, dtype=np.int16)

        # Apply mask
        segmentation_reshaped = segmentation_data.reshape(-1)
        nii_data_masked = nii_data_reshaped[segmentation_reshaped > 0, :]
        segmentation_masked = segmentation_reshaped[segmentation_reshaped > 0]

        if segment_labels is not None:
            segments = {}
            labels = list(segment_labels.keys())
            for label in labels:
                indices = np.array([], dtype=int)
                for ii in segment_labels[label]:
                    indices = np.concatenate([indices, np.where(segmentation_masked == ii)[0]])
                segments[label] = indices

    else:
        nii_data_masked = nii_data_reshaped

    bad_row_ind = np.where(~np.isfinite(nii_data_masked))[0]

    good_row_ind = np.ones(nii_data_masked.shape[0], dtype=bool)
    good_row_ind[bad_row_ind] = False

    nii_data_masked = nii_data_masked[good_row_ind, :]

    # Plot
    return nw_plot_carpet(
        nii_data_masked, detrend=detrend, segments=segments, output_file=output_file
    )


def get_segment_labels(filepath, keywords, delimiter=" ", index_position=0, label_position=1):
    """
    Return segment labels for plot_carpet function

    Parameters
    ----------
    filepath : :obj:`string`
        Path to segment label text file, such as freesurfer label file
    keywords : list of :obj:`string`
        List of label keywords. All labels containing the keyword will be grouped together.
        e.g. ["Cerebral_White_Matter", "Cerebral_Cortex", "Ventricle"]
    delimiter : :obj:`string`
        Delimiter between label index and label string in label file
        (' ' for freesurfer label file)
    index_position : :obj:`int`
        Position of label index in label file
        (0 for freesurfer label file)
    label_position : :obj:`int`
        Position of label string in label file
        (1 for freesurfer label file)

    Returns
    ---------
    dict
    e.g. {'Cerebral_White_Matter': [2, 41],
          'Cerebral_Cortex': [3, 42],
          'Ventricle': [4, 14, 15, 43, 72]}
    """
    segment_labels = {}

    with open(filepath, "r") as f:
        labels = f.read()

    labels_s = [label.split(delimiter) for label in labels.split("\n") if label != ""]

    for keyword in keywords:
        ind = [int(i[index_position]) for i in labels_s if keyword in i[label_position]]
        segment_labels[keyword] = ind

    return segment_labels
