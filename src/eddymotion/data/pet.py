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
"""PET data representation."""

from collections import namedtuple
from pathlib import Path
from tempfile import mkdtemp

import attr
import h5py
import nibabel as nb
import numpy as np
from nitransforms.linear import Affine


def _data_repr(value):
    if value is None:
        return "None"
    return f"<{'x'.join(str(v) for v in value.shape)} ({value.dtype})>"


@attr.s(slots=True)
class PET:
    """Data representation structure for PET data."""

    dataobj = attr.ib(default=None, repr=_data_repr)
    """A numpy ndarray object for the data array, without *b=0* volumes."""
    affine = attr.ib(default=None, repr=_data_repr)
    """Best affine for RAS-to-voxel conversion of coordinates (NIfTI header)."""
    brainmask = attr.ib(default=None, repr=_data_repr)
    """A boolean ndarray object containing a corresponding brainmask."""
    frame_time = attr.ib(default=None, repr=_data_repr)
    """A 1D numpy array with the midpoint timing of each sample."""
    total_duration = attr.ib(default=None, repr=_data_repr)
    """A float number representing the total duration of acquisition."""

    em_affines = attr.ib(default=None)
    """
    List of :obj:`nitransforms.linear.Affine` objects that bring
    PET timepoints into alignment.
    """
    _filepath = attr.ib(
        factory=lambda: Path(mkdtemp()) / "em_cache.h5",
        repr=False,
    )
    """A path to an HDF5 file to store the whole dataset."""

    def __len__(self):
        """Obtain the number of high-*b* orientations."""
        return self.dataobj.shape[-1]

    def set_transform(self, index, affine, order=3):
        """Set an affine, and update data object and gradients."""
        reference = namedtuple("ImageGrid", ("shape", "affine"))(
            shape=self.dataobj.shape[:3], affine=self.affine
        )
        xform = Affine(matrix=affine, reference=reference)

        if not Path(self._filepath).exists():
            self.to_filename(self._filepath)

        # read original PET
        with h5py.File(self._filepath, "r") as in_file:
            root = in_file["/0"]
            dframe = np.asanyarray(root["dataobj"][..., index])

        dmoving = nb.Nifti1Image(dframe, self.affine, None)

        # resample and update orientation at index
        self.dataobj[..., index] = np.asanyarray(
            xform.apply(dmoving, order=order).dataobj,
            dtype=self.dataobj.dtype,
        )

        # update transform
        if self.em_affines is None:
            self.em_affines = [None] * len(self)

        self.em_affines[index] = xform

    def to_filename(self, filename, compression=None, compression_opts=None):
        """Write an HDF5 file to disk."""
        filename = Path(filename)
        if not filename.name.endswith(".h5"):
            filename = filename.parent / f"{filename.name}.h5"

        with h5py.File(filename, "w") as out_file:
            out_file.attrs["Format"] = "EMC/PET"
            out_file.attrs["Version"] = np.uint16(1)
            root = out_file.create_group("/0")
            root.attrs["Type"] = "pet"
            for f in attr.fields(self.__class__):
                if f.name.startswith("_"):
                    continue

                value = getattr(self, f.name)
                if value is not None:
                    root.create_dataset(
                        f.name,
                        data=value,
                        compression=compression,
                        compression_opts=compression_opts,
                    )

    def to_nifti(self, filename, *_):
        """Write a NIfTI 1.0 file to disk."""
        nii = nb.Nifti1Image(self.dataobj, self.affine, None)
        nii.header.set_xyzt_units("mm")
        nii.to_filename(filename)

    @classmethod
    def from_filename(cls, filename):
        """Read an HDF5 file from disk."""
        with h5py.File(filename, "r") as in_file:
            root = in_file["/0"]
            data = {k: np.asanyarray(v) for k, v in root.items() if not k.startswith("_")}
        return cls(**data)


def load(
    filename,
    brainmask_file=None,
    frame_time=None,
    frame_duration=None,
):
    """Load PET data."""
    filename = Path(filename)
    if filename.name.endswith(".h5"):
        return PET.from_filename(filename)

    img = nb.load(filename)
    retval = PET(
        dataobj=img.get_fdata(dtype="float32"),
        affine=img.affine,
    )

    if frame_time is None:
        raise RuntimeError(
            "Start time of frames is mandatory (see https://bids-specification.readthedocs.io/"
            "en/stable/glossary.html#objects.metadata.FrameTimesStart)"
        )

    frame_time = np.array(frame_time, dtype="float32") - frame_time[0]
    if frame_duration is None:
        frame_duration = np.diff(frame_time)
        if len(frame_duration) == (retval.dataobj.shape[-1] - 1):
            frame_duration = np.append(frame_duration, frame_duration[-1])

    retval.total_duration = frame_time[-1] + frame_duration[-1]
    retval.frame_time = frame_time + 0.5 * np.array(frame_duration, dtype="float32")

    assert len(retval.frame_time) == retval.dataobj.shape[-1]

    if brainmask_file:
        mask = nb.load(brainmask_file)
        retval.brainmask = np.asanyarray(mask.dataobj)

    return retval
