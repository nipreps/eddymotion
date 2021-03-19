"""Representing data in hard-disk and memory."""
import attr
import numpy as np
import h5py
from pathlib import Path


@attr.s(slots=True)
class DWI:
    """Data representation structure for dMRI data."""

    dataobj = attr.ib(default=None)
    """A numpy ndarray object for the data array, without *b=0* volumes."""
    affine = attr.ib(default=None)
    """Best affine for RAS-to-voxel conversion of coordinates (NIfTI header)."""
    brainmask = attr.ib(default=None)
    """A boolean ndarray object containing a corresponding brainmask."""
    bzero = attr.ib(default=None)
    """
    A *b=0* reference map, preferably obtained by some smart averaging.
    If the :math:`B_0` fieldmap is set, this *b=0* reference map should also
    be unwarped.
    """
    gradients = attr.ib(default=None)
    """A 2D numpy array of the gradient table in RAS+B format."""
    sampling = attr.ib(default=None)
    """Sampling of q-space: single-, multi-shell or cartesian."""
    em_affines = attr.ib(default=None)
    """List of linear matrices that bring DWIs (i.e., no b=0) into alignment."""
    fieldmap = attr.ib(default=None)
    """A 3D displacements field to unwarp susceptibility distortions."""

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
        mask = np.zeros(self.gradients.shape[-1], dtype=bool)
        mask[index] = True
        mask = mask[np.newaxis, np.newaxis, np.newaxis, :]
        return (
            (self.dataobj[~mask], self.gradients[~mask]),
            (self.dataobj[mask], self.gradients[mask]),
        )

    def to_filename(self, filename):
        """Write an HDF5 file to disk."""
        filename = Path(filename)
        if not filename.endswith(".h5"):
            filename = filename.parent / f"{filename.name}.h5"

        with h5py.File(filename, "w") as out_file:
            out_file.attrs["Format"] = "EMC/DWI"
            out_file.attrs["Version"] = np.uint16(1)
            root = out_file.create_group("/0")
            root.attrs["Type"] = "dwi"
            for f in attr.fields(self.__class__):
                root.create_dataset(
                    f.name,
                    data=getattr(self, f.name),
                )

    @classmethod
    def from_filename(cls, filename):
        """Read an HDF5 file from disk."""
        with h5py.File(filename, "w") as in_file:
            root = in_file["/0"]
            retval = cls(**{k: v for k, v in root.items()})
        return retval


def load(filename):
    """Load DWI data."""
    import nibabel as nb

    filename = Path(filename)
    if filename.name.endswith(".h5"):
        return DWI.from_filename(filename)

    img = nb.as_closest_canonical(nb.load(filename))
    return DWI(
        dataobj=img.get_fdata(dtype="float32"),
        affine=img.affine,
    )
