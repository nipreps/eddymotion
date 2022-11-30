import ants
from eddymotion.dmri import DWI
import nibabel as nb
from eddymotion.utils import antsimage_from_path


def test_antsimage_from_path(datadir, tmp_path):
    """"""
    # Generate and save test data to file
    # Send through antsimage_from_path
    # Check if ANTsImage, same as original data
    fixed = tmp_path / "b0.nii.gz"
    dwdata = DWI.from_filename(datadir / "dwi.h5")
    b0nii = nb.Nifti1Image(dwdata.bzero, dwdata.affine, None)
    b0nii.to_filename(fixed)
    assert isinstance(antsimage_from_path(fixed), ants.ANTsImage)