"""A model-based algorithm for the realignment of dMRI data."""
from pathlib import Path
from tempfile import TemporaryDirectory, mkstemp
from pkg_resources import resource_filename as pkg_fn
from tqdm import tqdm
import numpy as np
import nibabel as nb
import nitransforms as nt
from nipype.interfaces.ants.registration import Registration
from eddymotion.model import ModelFactory


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
        dwdata : :obj:`~eddymotion.dmri.DWI`
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
            See :obj:`~eddymotion.model.ModelFactory` for allowed models (and corresponding
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
        align_kwargs = align_kwargs or {}
        reg_target_type = "dwi" if model.lower() not in ("b0", "s0") else "b0"

        if seed or seed == 0:
            np.random.seed(20210324 if seed is True else seed)

        bmask_img = None
        if dwdata.brainmask is not None:
            _, bmask_img = mkstemp(suffix="_bmask.nii.gz")
            nb.Nifti1Image(
                dwdata.brainmask.astype("uint8"), dwdata.affine, None
            ).to_filename(bmask_img)
            kwargs["mask"] = dwdata.brainmask

        kwargs["S0"] = _advanced_clip(dwdata.bzero)

        for i_iter in range(1, n_iter + 1):
            index_order = np.arange(len(dwdata))
            np.random.shuffle(index_order)
            with tqdm(total=len(index_order), unit="dwi") as pbar:
                for i in index_order:
                    # run a original-to-synthetic affine registration
                    with TemporaryDirectory() as tmpdir:
                        pbar.write(
                            f"Pass {i_iter}/{n_iter} | Processing b-index <{i}> in <{tmpdir}>"
                        )
                        data_train, data_test = dwdata.logo_split(i, with_b0=True)

                        # Factory creates the appropriate model and pipes arguments
                        dwmodel = ModelFactory.init(
                            gtab=data_train[1], model=model, **kwargs
                        )

                        # fit the model
                        dwmodel.fit(data_train[0])

                        # generate a synthetic dw volume for the test gradient
                        predicted = dwmodel.predict(data_test[1])

                        tmpdir = Path(tmpdir)
                        moving = tmpdir / "moving.nii.gz"
                        fixed = tmpdir / "fixed.nii.gz"
                        nb.Nifti1Image(
                            _advanced_clip(np.squeeze(data_test[0])),
                            dwdata.affine,
                            None,
                        ).to_filename(moving)
                        nb.Nifti1Image(
                            predicted
                            if model.lower() in ("b0", "s0")
                            else _advanced_clip(np.squeeze(predicted)),
                            dwdata.affine,
                            None,
                        ).to_filename(fixed)
                        registration = Registration(
                            terminal_output="file",
                            from_file=pkg_fn(
                                "eddymotion",
                                f"config/dwi-to-{reg_target_type}_level{i_iter}.json",
                            ),
                            fixed_image=str(fixed.absolute()),
                            moving_image=str(moving.absolute()),
                            **align_kwargs,
                        )
                        if bmask_img:
                            registration.inputs.fixed_image_masks = ["NULL", bmask_img]

                        if dwdata.em_affines and dwdata.em_affines[i] is not None:
                            mat_file = tmpdir / f"init{i_iter}.mat"
                            dwdata.em_affines[i].to_filename(mat_file, fmt="itk")
                            registration.inputs.initial_moving_transform = str(mat_file)

                        # execute ants command line
                        result = registration.run(cwd=str(tmpdir)).outputs

                        # read output transform
                        xform = nt.io.itk.ITKLinearTransform.from_filename(
                            result.forward_transforms[0]
                        ).to_ras(reference=fixed, moving=moving)

                    # update
                    dwdata.set_transform(i, xform)
                    pbar.update()

        return dwdata.em_affines


def _advanced_clip(
    data, p_min=35, p_max=99.98, nonnegative=True, dtype="int16", invert=False
):
    """
    Remove outliers at both ends of the intensity distribution and fit into a given dtype.

    This interface tries to emulate ANTs workflows' massaging that truncate images into
    the 0-255 range, and applies percentiles for clipping images.
    For image registration, normalizing the intensity into a compact range (e.g., uint8)
    is generally advised.

    To more robustly determine the clipping thresholds, spikes are removed from data with
    a median filter.
    Once the thresholds are calculated, the denoised data are thrown away and the thresholds
    are applied on the original image.

    """
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import ball

    # Calculate stats on denoised version, to preempt outliers from biasing
    denoised = ndimage.median_filter(data, footprint=ball(3))

    a_min = np.percentile(denoised[denoised > 0] if nonnegative else denoised, p_min)
    a_max = np.percentile(denoised[denoised > 0] if nonnegative else denoised, p_max)

    # Clip and cast
    data = np.clip(data, a_min=a_min, a_max=a_max)
    data -= data.min()
    data /= data.max()

    if invert:
        data = 1.0 - data

    if dtype in ("uint8", "int16"):
        data = np.round(255 * data).astype(dtype)

    return data
