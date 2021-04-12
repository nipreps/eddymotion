"""Representing data in hard-disk and memory."""
from pathlib import Path
from collections import namedtuple
from tempfile import mkdtemp
import attr
import numpy as np
import h5py
import nibabel as nb
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
    _filepath = attr.ib(default=Path(mkdtemp()) / "em_cache.h5", repr=False)
    """A path to an HDF5 file to store the whole dataset."""

    def __len__(self):
        """Obtain the number of high-*b* orientations."""
        return self.gradients.shape[-1]

    def logo_split(self, index, with_b0=False):
        """
        Produce one fold of LOGO (leave-one-gradient-out).

        Parameters
        ----------
        index : :obj:`int`
            Index of the DWI orientation to be left out in this fold.
        with_b0 : :obj:`bool`
            Insert the *b=0* reference at the beginning of the training dataset.

        Return
        ------
        (train_data, train_gradients) : :obj:`tuple`
            Training DWI and corresponding gradients.
            Training data/gradients come **from the updated dataset**.
        (test_data, test_gradients) :obj:`tuple`
            Test 3D map (one DWI orientation) and corresponding b-vector/value.
            The test data/gradient come **from the original dataset**.

        """
        if not Path(self._filepath).exists():
            self.to_filename(self._filepath)

        # read original DWI data & b-vector
        with h5py.File(self._filepath, "r") as in_file:
            root = in_file["/0"]
            dwframe = np.asanyarray(root["dataobj"][..., index])
            bframe = np.asanyarray(root["gradients"][..., index])

        # if the size of the mask does not match data, cache is stale
        mask = np.zeros(len(self), dtype=bool)
        mask[index] = True

        train_data = self.dataobj[..., ~mask]
        train_gradients = self.gradients[..., ~mask]

        if with_b0:
            train_data = np.concatenate(
                (np.asanyarray(self.bzero)[..., np.newaxis], train_data),
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
            (dwframe, bframe),
        )

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
            self.em_affines = [None] * len(self)

        self.em_affines[index] = xform

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

    def to_nifti(self, filename):
        """Write a NIfTI 1.0 file to disk."""
        nii = nb.Nifti1Image(
            self.dataobj,
            self.affine,
            None,
        )
        nii.header.set_xyzt_units("mm")
        nii.to_filename(filename)

    def plot_mosaic(self, index=None, **kwargs):
        """Visualize one direction of the dMRI dataset."""
        from eddymotion.viz import plot_dwi
        return plot_dwi(
            self.bzero if index is None else self.dataobj[..., index],
            self.affine,
            gradient=self.gradients[..., index] if index is not None else None,
            **kwargs,
        )

    def plot_gradients(self, **kwargs):
        """Visualize diffusion gradient."""
        from eddymotion.viz import plot_gradients
        return plot_gradients(
            self.gradients,
            **kwargs
        )

    @classmethod
    def from_filename(cls, filename):
        """Read an HDF5 file from disk."""
        with h5py.File(filename, "r") as in_file:
            root = in_file["/0"]
            data = {k: np.asanyarray(v) for k, v in root.items()}
        return cls(**data)


def load(
    filename, gradients_file=None, b0_file=None, brainmask_file=None, fmap_file=None
):
    """Load DWI data."""
    filename = Path(filename)
    if filename.name.endswith(".h5"):
        return DWI.from_filename(filename)

    if not gradients_file:
        raise RuntimeError("A gradients file is necessary")

    img = nb.as_closest_canonical(nb.load(filename))
    retval = DWI(
        affine=img.affine,
    )
    grad = np.loadtxt(gradients_file, dtype="float32").T
    gradmsk = grad[-1] > 50
    retval.gradients = grad[..., gradmsk]
    retval.dataobj = img.get_fdata(dtype="float32")[..., gradmsk]

    if b0_file:
        b0img = nb.as_closest_canonical(nb.load(b0_file))
        retval.bzero = np.asanyarray(b0img.dataobj)

    if brainmask_file:
        mask = nb.as_closest_canonical(nb.load(brainmask_file))
        retval.brainmask = np.asanyarray(mask.dataobj)

    if fmap_file:
        fmapimg = nb.as_closest_canonical(nb.load(fmap_file))
        retval.fieldmap = fmapimg.get_fdata(fmapimg, dtype="float32")

    return retval
