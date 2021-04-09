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


_epsi = 1.0e-9


def rotation_matrix(u, v):
    """
    Returns a rotation matrix R s.t. Ru = v.

    Extracted from the code of Emmanuel Caruyer
    (https://github.com/ecaruyer/qspace/blob/master/qspace/visu/visu_points.py).
    """
    # the axis is given by the product u x v
    u = u / np.sqrt((u ** 2).sum())
    v = v / np.sqrt((v ** 2).sum())
    w = np.asarray([u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0]])
    if (w ** 2).sum() < _epsi:
        # The vectors u and v are collinear
        return np.eye(3)

    # computes sine and cosine
    c = np.dot(u, v)
    s = np.sqrt((w ** 2).sum())

    w = w / s
    P = np.outer(w, w)
    Q = np.asarray([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = P + c * (np.eye(3) - P) + s * Q
    return R


def draw_circles(positions, rs):
    """
    Draw circular patches (lying on a sphere) at given positions.

    Adapted from the code of Emmanuel Caruyer
    (https://github.com/ecaruyer/qspace/blob/master/qspace/visu/visu_points.py)
    by the Nipreps developers.
    """
    # a circle centered at [1, 0, 0] with radius r
    M = 20
    t = np.linspace(0, 2 * np.pi, M)

    nbPoints = positions.shape[0]
    circles = np.zeros((nbPoints, M, 3))
    for i in range(positions.shape[0]):
        circleX = np.zeros((20, 3))
        dots_radius = np.sqrt(rs[i]) * 0.04
        circleX[:, 1] = dots_radius * np.cos(t)
        circleX[:, 2] = dots_radius * np.sin(t)
        norm = np.sqrt((positions[i] ** 2).sum())
        point = positions[i] / norm
        R1 = rotation_matrix(np.asarray([1, 0, 0]), point)
        circles[i] = positions[i] + np.dot(R1, circleX.T).T
    return circles


def draw_points(gradients, ax, colormap='viridis'):
    """
    Draw the vectors on a shell.

    Adapted from the code of Emmanuel Caruyer
    (https://github.com/ecaruyer/qspace/blob/master/qspace/visu/visu_points.py)
    by the Nipreps developers.

    Parameters
    ----------
    gradients : array-like shape (N, 4)
        A 2D numpy array of the gradient table in RAS+B format.

    ax : the matplolib axes instance to plot in.

    colormap : matplotlib colormap name
    """
    from matplotlib.pyplot import cm
    from mpl_toolkits.mplot3d import art3d

    # Initialize 3D view
    elev = 90
    azim = 0
    ax.view_init(azim=azim, elev=elev)

    # Normalize to 1 the highest bvalue
    bvals = np.copy(gradients[3, :])
    rs = bvals / bvals.max()

    # Color map depending on bvalue (for visualization)
    cmap = cm.get_cmap(colormap)
    colors = cmap(rs)

    # Relative shell radii proportional to the inverse of bvalue (for visualization)
    rs = np.reciprocal(rs)
    rs = rs / rs.max()
    rs = rs - rs.min()
    rs = rs / (rs.max() - rs.min())
    rs = rs * (0.7 - 0.4) + 0.4

    vects = np.copy(gradients[:3, :].T, )
    vects[vects[:, 2] < 0] *= -1

    # Render all gradient direction of all b-values
    circles = draw_circles(vects, rs)
    ax.add_collection(
        art3d.Poly3DCollection(
            circles,
            facecolors=colors,
            linewidth=0
        )
    )

    max_val = 0.6
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)
    ax.axis("off")


def plot_gradients(gradients, title="Shells reprojected", spacing=0.05, filename=None, **kwargs):
    """
    Draw the vectors on a unit sphere with color code for multiple b-value.

    Parameters
    ----------
    gradients : array-like shape (N, 4)
        A 2D numpy array of the gradient table in RAS+B format.

    title : :obj:`string`
        Custom plot title

    spacing : :obj:`float`
        Parameter to adjust plot spacing

    filename : :obj:`string`
        Path to save the plot

    kwargs : extra args given to :obj:`eddymotion.viz.draw_points()`
    """
    from matplotlib import pyplot as plt

    # Figure initialization
    fig = plt.figure(figsize=(9.0, 9.0))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=spacing,
                        top=1 - spacing,
                        wspace=2 * spacing)

    # Visualization after re-projecting all shells to the unit sphere
    draw_points(gradients, ax, **kwargs)

    # Add title
    plt.suptitle(title)

    # Save the figure if a filename is provided
    if filename is not None:
        plt.savefig(filename, dpi=200)
