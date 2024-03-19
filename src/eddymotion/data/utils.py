from pathlib import Path

import nibabel as nib
import nitransforms as nt
import numpy as np


def apply_affines(nii, em_affines, output_filename=None):
    """
    Apply affines to supplied nii data

    Parameters
    ----------
    nii : :obj:`Nifti1Image`
        Nifti1Image data to be transformed
    em_affines : :obj:`ndarray`
        Nx4x4 array
    output_filename : :obj:`str`, optional
        String specifying filepath to which to save transformed Nifti1Image data

    Returns
    -------
    nii_t_img : :obj:`Nifti1Image`
        Transformed Nifti1Image data

    """
    transformed_nii = np.zeros_like(np.asanyarray(nii.dataobj))

    for ii, bvecnii in enumerate(nib.four_to_three(nii)):
        xfms = nt.linear.Affine(em_affines[ii])
        transformed_nii[..., ii] = np.asanyarray((~xfms).apply(bvecnii, reference=nii).dataobj)

    nii_t_img = nii.__class__(transformed_nii, nii.affine, nii.header)

    if output_filename is not None:
        # Ensure directories in output_filename exist
        Path(output_filename).parent.mkdir(exist_ok=True)

        # Save as .nii
        nii_t_img.to_filename(output_filename)

    return nii_t_img
