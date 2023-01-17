import nibabel as nib
import numpy as np

from eddymotion.data.utils import apply_affines


def test_apply_affines():
    # Create synthetic dataset
    nii_data = np.random.rand(10, 10, 10, 10)

    # Generate Nifti1Image
    nii = nib.Nifti1Image(nii_data, np.eye(4))

    # Generate synthetic affines
    em_affines = np.concatenate([np.eye(4)[None]] * 3, 0)

    apply_affines(nii, em_affines)
