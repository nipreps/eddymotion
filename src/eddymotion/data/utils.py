import os
import os.path as op
from collections import namedtuple

import nibabel as nib
import nitransforms as nt


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
    transformed_nii : :obj:`Nifti1Image`
        Transformed Nifti1Image data

    """
    # Apply affine
    reference = namedtuple("ImageGrid", ("shape", "affine"))(
        shape=nii.dataobj.shape[:3], affine=nii.affine
    )

    xfms = nt.linear.LinearTransformsMapping(em_affines)
    transformed_nii = (~xfms).apply(nii, reference=reference)

    if output_filename is not None:
        # Ensure directories in output_filename exist
        os.makedirs(op.split(output_filename)[0], exist_ok=True)

        # Save as .nii
        nii_t_img = nib.Nifti1Image(transformed_nii, transformed_nii.affine)

        nib.save(nii_t_img, output_filename)

    return transformed_nii
