"""A SHORELine-like algorithm for the realignment of dMRI data."""
import attr
import numpy as np
from emc.utils.register import affine_registration


@attr.s(slots=True)
class EddyMotionEstimator:
    """Estimates rigid-body head-motion and distortions derived from eddy-currents."""

    dwdata = attr.ib(default=None)
    """A :class:`~emc.dmri.DWI` instance."""

    @staticmethod
    def fit(
        dwdata,
        *,
        n_iter=1,
        align_kwards=None,
        **kwargs,
    ):
        """Run the algorithm."""
        for _ in range(n_iter):
            for i in np.random.shuffle(range(len(dwdata))):
                data_train, data_test = dwdata.logo_split(i)

                # fit & predict
                model = ModelFactory(**kwargs).fit(
                    *data_train,
                    mask=dwdata.brainmask,
                )
                predicted = model.predict(
                    *data_test,
                    mask=dwdata.brainmask,
                )
                predicted[~dwdata.brainmask] = 0  # OE: very concerned about this

                # run volume-to-volume registration
                align_kwards = align_kwards or {}
                _, xform = affine_registration(
                    data_test[0],
                    predicted,
                    starting_affine=dwdata.em_affines[i],
                    **align_kwards,
                )

                # update
                dwdata.set_transform(i, xform)

        raise NotImplementedError

    def transform(
        *,
        X,
    ):
        """Generate a corrected NiBabel SpatialImage object."""
        raise NotImplementedError

        transform().to_filename("myfile.nii.gz")

    def fit_transform(
        *,
        X=None,
        init=None,
        **kwargs,
    ):
        """Execute both fitting and transforming."""
        raise NotImplementedError
