"""A model-based algorithm for the realignment of dMRI data."""
import numpy as np
from emc.utils.register import affine_registration
from emc.model import ModelFactory


class EddyMotionEstimator:
    """Estimates rigid-body head-motion and distortions derived from eddy-currents."""

    @staticmethod
    def fit(
        dwdata,
        *,
        n_iter=1,
        align_kwargs=None,
        model="b0",
        seed=None,
        **kwargs,
    ):
        r"""
        Estimate head-motion and Eddy currents.

        Parameters
        ----------
        dwdata : :obj:`~emc.dmri.DWI`
            The target DWI dataset, represented by this tool's internal
            type. The object is used in-place, and will contain the estimated
            parameters in its ``em_affines`` property, as well as the rotated
            *b*-vectors within its ``gradients`` property.
        n_iter : :obj:`int`
            Number of iterations this particular model is going to be repeated.
        align_kwargs : :obj:`dict`
            Parameters to configure the image registration process.
        model : :obj:`str`
            Selects the diffusion model that will generate the registration target
            corresponding to each gradient map.
            See :obj:`~emc.model.ModelFactory` for allowed models (and corresponding
            keywords).
        seed : :obj:`int` or :obj:`bool`
            Seed the random number generator (necessary when we want deterministic
            estimation).

        Return
        ------
        affines : :obj:`list` of :obj:`numpy.ndarray`
            A list of :math:`4 \times 4` affine matrices encoding the estimated
            parameters of the deformations caused by head-motion and eddy-currents.

        """
        if seed or seed == 0:
            np.random.seed(20210324 if seed is True else seed)

        for _ in range(n_iter):
            for i in np.random.shuffle(range(len(dwdata))):
                data_train, data_test = dwdata.logo_split(i)

                # fit the diffusion model
                model = ModelFactory.init(gtab=data_train[1], model=model).fit(
                    *data_train,
                    mask=dwdata.brainmask,
                    **kwargs,
                )

                # generate a synthetic gradient volume
                predicted = model.predict(
                    *data_test,
                    mask=dwdata.brainmask,
                    S0=dwdata.bzero,
                    **kwargs,
                )

                # run a original-to-synthetic affine registration
                init_affine = dwdata.em_affines[i] if dwdata.em_affines else None
                align_kwargs = align_kwargs or {}
                _, xform = affine_registration(
                    data_test[0],  # moving
                    predicted,  # fixed
                    starting_affine=init_affine,
                    **align_kwargs,
                )

                # update
                dwdata.set_transform(i, xform)

        return dwdata.em_affines
