from pathlib import Path, PurePosixPath
import nibabel as nb
import ants


def antsimage_from_path(val):
    """
    Returns ANTsImage from .nii or .gz path.

    Parameters
    ----------
    val : 
        Value in dictionary of registration arguments

    Returns
    -------
    :obj:`ANTsImage` if val is a string corresponding to a path ending in .nii.gz or .nii
    """
    if isinstance(val, str):
        if Path(val).exists() and (PurePosixPath(val).suffix == '.nii.gz' or PurePosixPath(val).suffix == '.nii'):
            return ants.from_nibabel(nb.load(val))
    
    return val