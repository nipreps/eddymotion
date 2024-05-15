import nibabel as nib
import numpy as np
import numpy.testing as npt

from eddymotion.data.utils import apply_affines


def test_apply_affines():
    # Create synthetic dataset
    nii_data = np.random.rand(10, 10, 10, 10)

    # Generate Nifti1Image
    nii = nib.Nifti1Image(nii_data, np.eye(4))

    # Generate synthetic affines
    em_affines = np.expand_dims(np.eye(4), 0).repeat(nii_data.shape[-1], 0)

    nii_t = apply_affines(nii, em_affines)

    npt.assert_allclose(nii.dataobj, nii_t.dataobj)
    npt.assert_array_equal(nii.affine, nii_t.affine)
