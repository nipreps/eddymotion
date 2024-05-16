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
"""Representing data in hard-disk and memory."""

from collections import namedtuple
from pathlib import Path
from tempfile import mkdtemp
from warnings import warn

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
class DWI:
    """Data representation structure for dMRI data."""

    dataobj = attr.ib(default=None, repr=_data_repr)
    """A numpy ndarray object for the data array, without *b=0* volumes."""
    affine = attr.ib(default=None, repr=_data_repr)
    """Best affine for RAS-to-voxel conversion of coordinates (NIfTI header)."""
    brainmask = attr.ib(default=None, repr=_data_repr)
    """A boolean ndarray object containing a corresponding brainmask."""
    bzero = attr.ib(default=None, repr=_data_repr)
    """
    A *b=0* reference map, preferably obtained by some smart averaging.
    If the :math:`B_0` fieldmap is set, this *b=0* reference map should also
    be unwarped.
    """
    gradients = attr.ib(default=None, repr=_data_repr)
    """A 2D numpy array of the gradient table in RAS+B format."""
    em_affines = attr.ib(default=None)
    """
    List of :obj:`nitransforms.linear.Affine` objects that bring
    DWIs (i.e., no b=0) into alignment.
    """
    fieldmap = attr.ib(default=None, repr=_data_repr)
    """A 3D displacements field to unwarp susceptibility distortions."""
    _filepath = attr.ib(
        factory=lambda: Path(mkdtemp()) / "em_cache.h5",
        repr=False,
    )
    """A path to an HDF5 file to store the whole dataset."""

    def get_filename(self):
        """Get the filepath of the HDF5 file."""
        return self._filepath

    def __len__(self):
        """Obtain the number of high-*b* orientations."""
        return self.dataobj.shape[-1]

    def set_transform(self, index, affine, order=3):
        """Set an affine, and update data object and gradients."""
        reference = namedtuple("ImageGrid", ("shape", "affine"))(
            shape=self.dataobj.shape[:3], affine=self.affine
        )

        # create a nitransforms object
        if self.fieldmap:
            # compose fieldmap into transform
            raise NotImplementedError
        else:
            xform = Affine(matrix=affine, reference=reference)

        if not Path(self._filepath).exists():
            self.to_filename(self._filepath)

        # read original DWI data & b-vector
        with h5py.File(self._filepath, "r") as in_file:
            root = in_file["/0"]
            dwframe = np.asanyarray(root["dataobj"][..., index])
            bvec = np.asanyarray(root["gradients"][:3, index])

        dwmoving = nb.Nifti1Image(dwframe, self.affine, None)

        # resample and update orientation at index
        self.dataobj[..., index] = np.asanyarray(
            xform.apply(dwmoving, order=order).dataobj,
            dtype=self.dataobj.dtype,
        )

        # invert transform transform b-vector and origin
        r_bvec = (~xform).map([bvec, (0.0, 0.0, 0.0)])
        # Reset b-vector's origin
        new_bvec = r_bvec[1] - r_bvec[0]
        # Normalize and update
        self.gradients[:3, index] = new_bvec / np.linalg.norm(new_bvec)

        # update transform
        if self.em_affines is None:
            self.em_affines = np.zeros((self.dataobj.shape[-1], 4, 4))

        self.em_affines[index] = xform.matrix

    def to_filename(self, filename, compression=None, compression_opts=None):
        """Write an HDF5 file to disk."""
        filename = Path(filename)
        if not filename.name.endswith(".h5"):
            filename = filename.parent / f"{filename.name}.h5"

        with h5py.File(filename, "w") as out_file:
            out_file.attrs["Format"] = "EMC/DWI"
            out_file.attrs["Version"] = np.uint16(1)
            root = out_file.create_group("/0")
            root.attrs["Type"] = "dwi"
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

    def to_nifti(self, filename, **kwargs):
        """Write a NIfTI 1.0 file to disk."""
        insert_b0 = kwargs.get("insert_b0", False)
        data = (
            self.dataobj
            if not insert_b0
            else np.concatenate((self.bzero[..., np.newaxis], self.dataobj), axis=-1)
        )
        nii = nb.Nifti1Image(data, self.affine, None)
        nii.header.set_xyzt_units("mm")
        nii.to_filename(filename)

    def plot_mosaic(self, index=None, **kwargs):
        """Visualize one direction of the dMRI dataset."""
        from nireports.reportlets.modality.dwi import plot_dwi

        return plot_dwi(
            self.bzero if index is None else self.dataobj[..., index],
            self.affine,
            gradient=self.gradients[..., index] if index is not None else None,
            **kwargs,
        )

    def plot_gradients(self, **kwargs):
        """Visualize diffusion gradient."""
        from nireports.reportlets.modality.dwi import plot_gradients as rpt_plot_gradients

        return rpt_plot_gradients(self.gradients, **kwargs)

    @classmethod
    def from_filename(cls, filename):
        """Read an HDF5 file from disk."""
        with h5py.File(filename, "r") as in_file:
            root = in_file["/0"]
            data = {k: np.asanyarray(v) for k, v in root.items() if not k.startswith("_")}
        return cls(**data)


def load(
    filename,
    gradients_file=None,
    b0_file=None,
    brainmask_file=None,
    fmap_file=None,
    bvec_file=None,
    bval_file=None,
    b0_thres=50,
):
    """Load DWI data."""
    filename = Path(filename)
    if filename.name.endswith(".h5"):
        return DWI.from_filename(filename)

    if gradients_file:
        grad = np.loadtxt(gradients_file, dtype="float32").T

        if bvec_file and bval_file:
            warn(
                "Gradients table file and b-vec/val files are defined; "
                "dismissing b-vec/val files.",
                stacklevel=2,
            )
    elif bvec_file and bval_file:
        grad = np.vstack(
            (
                np.loadtxt(bvec_file, dtype="float32"),
                np.loadtxt(bval_file, dtype="float32"),
            )
        )
    else:
        raise RuntimeError("A gradients file is necessary")

    img = nb.load(filename)
    fulldata = img.get_fdata(dtype="float32")
    retval = DWI(
        affine=img.affine,
    )
    gradmsk = grad[-1] > b0_thres
    retval.gradients = grad[..., gradmsk]
    retval.dataobj = fulldata[..., gradmsk]

    if b0_file:
        b0img = nb.load(b0_file)
        retval.bzero = np.asanyarray(b0img.dataobj)
    elif not np.all(gradmsk):
        retval.bzero = np.median(fulldata[..., ~gradmsk], axis=3)

    if brainmask_file:
        mask = nb.load(brainmask_file)
        retval.brainmask = np.asanyarray(mask.dataobj)

    if fmap_file:
        fmapimg = nb.load(fmap_file)
        retval.fieldmap = fmapimg.get_fdata(fmapimg, dtype="float32")

    return retval
