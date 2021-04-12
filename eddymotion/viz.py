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
    affine[:3, 3] = -1.0 * (affine[:3, :3] @ ((np.array(dataobj.shape) - 1) * 0.5))

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




def rotation_matrix(u, v):
    r"""
    Calculate the rotation matrix *R* such that :math:`R \cdot \mathbf{u} = \mathbf{v}`.

    Extracted from `Emmanuel Caruyer's code
    <https://github.com/ecaruyer/qspace/blob/master/qspace/visu/visu_points.py>`__,
    which is distributed under the revised BSD License:

    Copyright (c) 2013-2015, Emmanuel Caruyer
    All rights reserved.

    .. admonition :: List of changes

        Only minimal updates to leverage Numpy.

    Parameters
    ----------
    u : :obj:`numpy.ndarray`
        A vector.
    v : :obj:`numpy.ndarray`
        A vector.

    Returns
    -------
    R : :obj:`numpy.ndarray`
        The rotation matrix.

    """
    # the axis is given by the product u x v
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = np.asarray(
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    )
    if (w ** 2).sum() < (np.finfo(w.dtype).eps * 10):
        # The vectors u and v are collinear
        return np.eye(3)

    # computes sine and cosine
    c = u @ v
    s = np.linalg.norm(w)

    w = w / s
    P = np.outer(w, w)
    Q = np.asarray([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = P + c * (np.eye(3) - P) + s * Q
    return R


def draw_circles(positions, radius, n_samples=20):
    r"""
    Draw circular patches (lying on a sphere) at given positions.

    Adapted from from `Emmanuel Caruyer's code
    <https://github.com/ecaruyer/qspace/blob/master/qspace/visu/visu_points.py>`__,
    which is distributed under the revised BSD License:

    Copyright (c) 2013-2015, Emmanuel Caruyer
    All rights reserved.

    .. admonition :: List of changes

        Modified to take the full list of normalized bvecs and corresponding circle
        radii instead of taking the list of bvecs and radii for a specific shell
        (*b*-value).

    Parameters
    ----------
    positions : :obj:`numpy.ndarray`
        An array :math:`N \times 3` of 3D cartesian positions.
    radius : :obj:`float`
        The reference radius (or, the radius in single-shell plots)
    n_samples : :obj:`int`
        The number of samples on the sphere.

    Returns
    -------
    circles : :obj:`numpy.ndarray`
        Circular patches

    """
    # a circle centered at [1, 0, 0] with radius r
    t = np.linspace(0, 2 * np.pi, n_samples)

    nb_points = positions.shape[0]
    circles = np.zeros((nb_points, n_samples, 3))
    for i in range(positions.shape[0]):
        circle_x = np.zeros((n_samples, 3))
        dots_radius = np.sqrt(radius[i]) * 0.04
        circle_x[:, 1] = dots_radius * np.cos(t)
        circle_x[:, 2] = dots_radius * np.sin(t)
        norm = np.linalg.norm(positions[i])
        point = positions[i] / norm
        r1 = rotation_matrix(np.asarray([1, 0, 0]), point)
        circles[i] = positions[i] + np.dot(r1, circle_x.T).T
    return circles


def draw_points(gradients, ax, rad_min=0.3, rad_max=0.7, cmap="viridis"):
    """
    Draw the vectors on a shell.

    Adapted from from `Emmanuel Caruyer's code
    <https://github.com/ecaruyer/qspace/blob/master/qspace/visu/visu_points.py>`__,
    which is distributed under the revised BSD License:

    Copyright (c) 2013-2015, Emmanuel Caruyer
    All rights reserved.

    .. admonition :: List of changes

        * The input is a single 2D numpy array of the gradient table in RAS+B format
        * The scaling of the circle radius for each bvec proportional to the inverse of
          the bvals. A minimum/maximal value for the radii can be specified.
        * Circles for each bvec are drawn at once instead of looping over the shells.
        * Some variables have been renamed (like vects to bvecs)

    Parameters
    ----------
    gradients : array-like shape (N, 4)
        A 2D numpy array of the gradient table in RAS+B format.
    ax : :obj:`matplotlib.axes.Axis`
        The matplolib axes instance to plot in.
    rad_min : :obj:`float` between 0 and 1
        Minimum radius of the circle that renders a gradient direction
    rad_max : :obj:`float` between 0 and 1
        Maximum radius of the circle that renders a gradient direction
    cmap : :obj:`matplotlib.pyplot.cm.ColorMap`
        matplotlib colormap name

    """
    from matplotlib.pyplot import cm
    from mpl_toolkits.mplot3d import art3d

    # Initialize 3D view
    elev = 90
    azim = 0
    ax.view_init(azim=azim, elev=elev)

    # Normalize to 1 the highest bvalue
    bvals = np.copy(gradients[3, :])
    bvals = bvals / bvals.max()

    if isinstance(cmap, (str, bytes)):
        cmap = cm.get_cmap(cmap)

    # Color map depending on bvalue (for visualization)
    colors = cmap(bvals)

    # Relative shell radii proportional to the inverse of bvalue (for visualization)
    rs = np.reciprocal(bvals)
    rs = rs / rs.max()

    # Readjust radius of the circle given the minimum and maximal allowed values.
    rs = rs - rs.min()
    rs = rs / (rs.max() - rs.min())
    rs = rs * (rad_max - rad_min) + rad_min

    bvecs = np.copy(
        gradients[:3, :].T,
    )
    bvecs[bvecs[:, 2] < 0] *= -1

    # Render all gradient direction of all b-values
    circles = draw_circles(bvecs, rs)
    ax.add_collection(art3d.Poly3DCollection(circles, facecolors=colors, linewidth=0))

    max_val = 0.6
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)
    ax.axis("off")


def plot_gradients(
    gradients,
    title=None,
    ax=None,
    spacing=0.05,
    filename=None,
    **kwargs,
):
    """
    Draw the vectors on a unit sphere with color code for multiple b-value.

    Parameters
    ----------
    gradients : array-like shape (N, 4)
        A 2D numpy array of the gradient table in RAS+B format.
    title : :obj:`string`
        Custom plot title
    ax : :obj:`matplotlib.axes.Axis`
        A figure's axis to plot on.
    spacing : :obj:`float`
        Parameter to adjust plot spacing
    filename : :obj:`string`
        Path to save the plot
    kwargs : extra args given to :obj:`eddymotion.viz.draw_points()`

    """
    from matplotlib import pyplot as plt

    # Figure initialization
    if ax is None:
        figsize = kwargs.pop("figsize", (9.0, 9.0))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=spacing, top=1 - spacing, wspace=2 * spacing)

    # Visualization after re-projecting all shells to the unit sphere
    draw_points(gradients, ax, **kwargs)

    if title:
        plt.suptitle(title)

    return ax
