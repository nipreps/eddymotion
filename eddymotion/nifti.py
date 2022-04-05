"""Function to save object in nifti format"""

def _to_nifti(data, affine, filename, clip=True):
    data = np.squeeze(data)
    if clip:
        data = _advanced_clip(data)
    nb.Nifti1Image(
        data,
        affine,
        None,
    ).to_filename(filename)
