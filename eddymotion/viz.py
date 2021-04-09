"""Visualization utilities."""
import numpy as np
import nibabel as nb


def plot_dwi(dataobj, affine, gradient=None, **kwargs):
    """Plot a DW map."""
    from nilearn.plotting import plot_anat
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
        }
    )

    affine = np.diag(nb.affines.voxel_sizes(affine).tolist() + [1])
    affine[:3, 3] = -1.0 * (
        affine[:3, :3] @ ((np.array(dataobj.shape) - 1) * 0.5)
    )

    vmax = kwargs.pop("vmax", None) or np.percentile(dataobj, 98)
    cut_coords = kwargs.pop("cut_coords", None) or (0, 0, 0)

    return plot_anat(
        nb.Nifti1Image(dataobj, affine, None),
        vmax=vmax,
        cut_coords=cut_coords,
        title=r"Reference $b$=0"
        if gradient is None
        else f"""\
$b$={gradient[3].astype(int)}, \
$\\vec{{b}}$ = ({', '.join(str(v) for v in gradient[:3])})""",
        **kwargs,
    )
