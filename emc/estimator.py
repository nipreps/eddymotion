"""A SHORELine-like algorithm for the realignment of dMRI data."""
import attr
import numpy as np
from pathlib import Path


@attr.s(slots=True)
class EddyMotionEstimator:
    """Estimates rigid-body head-motion and distortions derived from eddy-currents."""

    datapath = attr.ib(default=None)
    """Path to the input DWI dataset."""
    gradients = attr.ib(default=None)
    """A 2D numpy array of the gradient table in RAS+B format."""
    bzero = attr.ib(default=None)
    """A *b=0* reference map, preferably obtained by some smart averaging."""
    model = attr.ib(default="SFM", type="str")
    """The prediction model - options: SFM, SHORE, Tensor, DKI."""
    transforms = attr.ib(default=None)
    """Current transform parameters (list of affine matrices)."""

    def fit(
        *,
        X=None,
        init=None,
        **kwargs,
    ):
        """
        Run the algorithm.

        Parameters
        ----------
        X : :obj:`~nibabel.SpatialImage`
          A DWI image, as a nibabel image object.
        init : :obj:`~numpy.ndarray`
          An Nx4x4 array of initialization transforms.

        """
        loo_index = np.random.shuffle(range(len(X)))

        self.transforms = [
            _emc(X, index=i)
            for i in loo_index
        ]
        raise NotImplementedError

    def predict():
        """Generate a corrected NiBabel SpatialImage object."""


def _emc(x, index, model="SFM"):
    return NotImplementedError
