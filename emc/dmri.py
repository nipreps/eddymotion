"""Representing data in hard-disk and memory."""
from collections import namedtuple
import attr
import numpy as np
import h5py
from pathlib import Path
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
    """List of linear matrices that bring DWIs (i.e., no b=0) into alignment."""
    fieldmap = attr.ib(default=None, repr=_data_repr)
    """A 3D displacements field to unwarp susceptibility distortions."""
    _filepath = attr.ib(default=None)
    """A path to an HDF5 file to store the whole dataset."""

    def __len__(self):
        """Obtain the number of high-*b* orientations."""
        return self.gradients.shape[-1]

    def logo_split(self, index):
        """
        Produce one fold of LOGO (leave-one-gradient-out).

        Parameters
        ----------
        index : :obj:`int`
            Index of the DWI orientation to be left out in this fold.

        Return
        ------
        (train_data, train_gradients) : :obj:`tuple`
            Training DWI and corresponding gradients
        (test_data, test_gradients) :obj:`tuple`
            Test 3D map (one DWI orientation) and corresponding b-vector/value.

        """
        mask = np.zeros(len(self), dtype=bool)
        mask[index] = True
        return (
            (self.dataobj[..., ~mask], self.gradients[..., ~mask]),
            (self.dataobj[..., mask], self.gradients[..., mask]),
        )

    def set_transform(self, index, affine, order=1):
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

        # read original DWI data & b-vector
        with h5py.File(self.filepath, "r") as in_file:
            root = in_file["/0"]
            dwframe = root["dataobj"][..., index]
            bvec = root["gradients"][:3, index]

        # resample and update orientation at index
        self.dataobj[..., index] = xform.apply(dwframe)

        # invert transform transform b-vector and origin
        r_bvec = ~xform.apply([bvec, (0.0, 0.0, 0.0)])
        # Reset b-vector's origin
        new_bvec = r_bvec[1] - r_bvec[0]
        # Normalize and update
        self.gradients[:3, index] = new_bvec / np.norm(new_bvec)

        # update transform
        if self.em_affines is None:
            self.em_affines = [np.eye(4)] * len(self)

        self.em_affines[index] = xform

    def to_filename(self, filename):
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
                    )
        self._filepath = filename

    @classmethod
    def from_filename(cls, filename):
        """Read an HDF5 file from disk."""
        with h5py.File(filename, "r") as in_file:
            root = in_file["/0"]
            retval = cls(**{k: v for k, v in root.items()})
        return retval


def load(filename, gradients_file, b0_file=None, fmap_file=None):
    """Load DWI data."""
    import nibabel as nb

    filename = Path(filename)
    if filename.name.endswith(".h5"):
        return DWI.from_filename(filename)

    img = nb.as_closest_canonical(nb.load(filename))
    retval = DWI(
        affine=img.affine,
    )
    grad = np.loadtxt(gradients_file).T
    gradmsk = grad[-1] > 50
    retval.gradients = grad[..., gradmsk]
    retval.dataobj = img.get_fdata(dtype="float32")[..., gradmsk]

    if b0_file:
        b0img = nb.as_closest_canonical(nb.load(b0_file))
        retval.bzero = np.asanyarray(b0img.dataobj)

    if fmap_file:
        fmapimg = nb.as_closest_canonical(nb.load(fmap_file))
        retval.fieldmap = fmapimg.get_fdata(fmapimg, dtype="float32")

    return retval
