# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""A model-based algorithm for the realignment of dMRI data."""

from collections import namedtuple
from pathlib import Path
from tempfile import TemporaryDirectory, mkstemp

import nibabel as nb
import nitransforms as nt
import numpy as np
from nipype.interfaces.ants.registration import Registration
from nitransforms.linear import Affine
from pkg_resources import resource_filename as pkg_fn
from tqdm import tqdm

from eddymotion.data.splitting import lovo_split
from eddymotion.model import ModelFactory


class EddyMotionEstimator:
    """Estimates rigid-body head-motion and distortions derived from eddy-currents."""

    @staticmethod
    def estimate(
        dwdata,
        *,
        align_kwargs=None,
        models=("b0",),
        omp_nthreads=None,
        n_jobs=None,
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
        models : :obj:`list`
            Selects the diffusion model that will generate the registration target
            corresponding to each gradient map.
            See :obj:`~eddymotion.model.ModelFactory` for allowed models (and corresponding
            keywords).
        omp_nthreads : :obj:`int`
            Maximum number of threads an individual process may use.
        n_jobs : :obj:`int`
            Number of parallel jobs.
        seed : :obj:`int` or :obj:`bool`
            Seed the random number generator (necessary when we want deterministic
            estimation). See :func:`_sort_dwdata_indices`.
        Return
        ------
        :obj:`list` of :obj:`numpy.ndarray`
            A list of :math:`4 \times 4` affine matrices encoding the estimated
            parameters of the deformations caused by head-motion and eddy-currents.

        """

        align_kwargs = align_kwargs or {}

        index_order = _sort_dwdata_indices(seed, len(dwdata))

        if "num_threads" not in align_kwargs and omp_nthreads is not None:
            align_kwargs["num_threads"] = omp_nthreads

        n_iter = len(models)
        for i_iter, model in enumerate(models):
            reg_target_type = (
                "dwi" if model.lower() not in ("b0", "s0", "avg", "average", "mean") else "b0"
            )

            # When downsampling these need to be set per-level
            bmask_img = _prepare_brainmask_data(dwdata.brainmask, dwdata.affine)

            _prepare_kwargs(dwdata, kwargs)

            single_model = model.lower() in (
                "b0",
                "s0",
                "avg",
                "average",
                "mean",
            ) or model.lower().startswith("full")

            dwmodel = None
            if single_model:
                if model.lower().startswith("full"):
                    model = model[4:]

                # Factory creates the appropriate model and pipes arguments
                dwmodel = ModelFactory.init(
                    model=model,
                    **kwargs,
                )
                dwmodel.fit(dwdata.dataobj, n_jobs=n_jobs)

            with TemporaryDirectory() as tmp_dir:
                print(f"Processing in <{tmp_dir}>")
                ptmp_dir = Path(tmp_dir)
                with tqdm(total=len(index_order), unit="dwi") as pbar:
                    # run a original-to-synthetic affine registration
                    for i in index_order:
                        pbar.set_description_str(
                            f"Pass {i_iter + 1}/{n_iter} | Fit and predict b-index <{i}>"
                        )
                        data_train, data_test = lovo_split(dwdata, i, with_b0=True)
                        grad_str = f"{i}, {data_test[1][:3]}, b={int(data_test[1][3])}"
                        pbar.set_description_str(f"[{grad_str}], {n_jobs} jobs")

                        if not single_model:  # A true LOGO estimator
                            if hasattr(dwdata, "gradients"):
                                kwargs["gtab"] = data_train[1]
                            # Factory creates the appropriate model and pipes arguments
                            dwmodel = ModelFactory.init(
                                model=model,
                                n_jobs=n_jobs,
                                **kwargs,
                            )

                            # fit the model
                            dwmodel.fit(
                                data_train[0],
                                n_jobs=n_jobs,
                            )

                        # generate a synthetic dw volume for the test gradient
                        predicted = dwmodel.predict(data_test[1])

                        # prepare data for running ANTs
                        fixed, moving = _prepare_registration_data(
                            data_test[0], predicted, dwdata.affine, i, ptmp_dir, reg_target_type
                        )

                        pbar.set_description_str(
                            f"Pass {i_iter + 1}/{n_iter} | Realign b-index <{i}>"
                        )

                        xform = _run_registration(
                            fixed,
                            moving,
                            bmask_img,
                            dwdata.em_affines,
                            dwdata.affine,
                            dwdata.dataobj.shape[:3],
                            data_test[1][3],
                            dwdata.fieldmap,
                            i_iter,
                            i,
                            ptmp_dir,
                            reg_target_type,
                            align_kwargs,
                        )

                        # update
                        dwdata.set_transform(i, xform.matrix)
                        pbar.update()

        return dwdata.em_affines


def _advanced_clip(data, p_min=35, p_max=99.98, nonnegative=True, dtype="int16", invert=False):
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


def _to_nifti(data, affine, filename, clip=True):
    data = np.squeeze(data)
    if clip:
        data = _advanced_clip(data)
    nii = nb.Nifti1Image(
        data,
        affine,
        None,
    )
    nii.header.set_sform(affine, code=1)
    nii.header.set_qform(affine, code=1)
    nii.to_filename(filename)


def _sort_dwdata_indices(seed, dwi_vol_count):
    """Sort the DWI data volume indices.

    Parameters
    ----------
    seed : :obj:`int` or :obj:`bool`
        Seed the random number generator. If an integer, the value is used to initialize the
        generator; if ``True``, the arbitrary value of ``20210324`` is used to initialize it.
    dwi_vol_count : :obj:`int`
        Number of DWI volumes.

    Returns
    -------
    index_order : :obj:`numpy.ndarray`
        Index order.
    """

    _seed = None
    if seed or seed == 0:
        _seed = 20210324 if seed is True else seed

    rng = np.random.default_rng(_seed)

    index_order = np.arange(dwi_vol_count)
    rng.shuffle(index_order)

    return index_order


def _prepare_brainmask_data(brainmask, affine):
    """Prepare the brainmask data: save the data to disk.

    Parameters
    ----------
    brainmask : :obj:`numpy.ndarray`
        Brainmask data.
    affine : :obj:`numpy.ndarray`
        Affine transformation matrix.

    Returns
    -------
    bmask_img : :class:`~nibabel.nifti1.Nifti1Image`
        Brainmask image.
    """

    bmask_img = None
    if brainmask is not None:
        _, bmask_img = mkstemp(suffix="_bmask.nii.gz")
        nb.Nifti1Image(brainmask.astype("uint8"), affine, None).to_filename(bmask_img)
    return bmask_img


def _prepare_kwargs(dwdata, kwargs):
    """Prepare the keyword arguments depending on the DWI data: add attributes corresponding to
    the ``brainmask``, ``bzero``, ``gradients``, ``frame_time``, and ``total_duration`` DWI data
    properties.

    Modifies kwargs in-place.

    Parameters
    ----------
    dwdata : :class:`eddymotion.data.dmri.DWI`
        DWI data object.
    kwargs : :obj:`dict`
        Keyword arguments.
    """

    if dwdata.brainmask is not None:
        kwargs["mask"] = dwdata.brainmask

    if hasattr(dwdata, "bzero") and dwdata.bzero is not None:
        kwargs["S0"] = _advanced_clip(dwdata.bzero)

    if hasattr(dwdata, "gradients"):
        kwargs["gtab"] = dwdata.gradients

    if hasattr(dwdata, "frame_time"):
        kwargs["timepoints"] = dwdata.frame_time

    if hasattr(dwdata, "total_duration"):
        kwargs["xlim"] = dwdata.total_duration


def _prepare_registration_data(dwframe, predicted, affine, vol_idx, dirname, reg_target_type):
    """Prepare the registration data: save the fixed and moving images to disk.

    Parameters
    ----------
    dwframe : :obj:`numpy.ndarray`
        DWI data object.
    predicted : :obj:`numpy.ndarray`
        Predicted data.
    affine : :obj:`numpy.ndarray`
        Affine transformation matrix.
    vol_idx : :obj:`int
        DWI volume index.
    dirname : :obj:`Path`
        Directory name where the data is saved.
    reg_target_type : :obj:`str`
        Target registration type.

    Returns
    -------
    fixed : :obj:`Path`
        Fixed image filename.
    moving : :obj:`Path`
        Moving image filename.
    """

    moving = dirname / f"moving{vol_idx:05d}.nii.gz"
    fixed = dirname / f"fixed{vol_idx:05d}.nii.gz"
    _to_nifti(dwframe, affine, moving)
    _to_nifti(
        predicted,
        affine,
        fixed,
        clip=reg_target_type == "dwi",
    )
    return fixed, moving


def _run_registration(
    fixed,
    moving,
    bmask_img,
    em_affines,
    affine,
    shape,
    bval,
    fieldmap,
    i_iter,
    vol_idx,
    dirname,
    reg_target_type,
    align_kwargs,
):
    """Register the moving image to the fixed image.

    Parameters
    ----------
    fixed : :obj:`Path`
        Fixed image filename.
    moving : :obj:`Path`
        Moving image filename.
    bmask_img : :class:`~nibabel.nifti1.Nifti1Image`
        Brainmask image.
    em_affines : :obj:`numpy.ndarray`
        Estimated eddy motion affine transformation matrices.
    affine : :obj:`numpy.ndarray`
        Affine transformation matrix.
    shape : :obj:`tuple`
        Shape of the DWI frame.
    bval : :obj:`int`
        b-value of the corresponding DWI volume.
    fieldmap : :class:`~nibabel.nifti1.Nifti1Image`
        Fieldmap.
    i_iter : :obj:`int`
        Iteration number.
    vol_idx : :obj:`int`
        DWI frame index.
    dirname : :obj:`Path`
        Directory name where the transformation is saved.
    reg_target_type : :obj:`str`
        Target registration type.
    align_kwargs : :obj:`dict`
        Parameters to configure the image registration process.

    Returns
    -------
    xform : :class:`~nitransforms.linear.Affine`
        Registration transformation.
    """

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

    if em_affines is not None and np.any(em_affines[vol_idx, ...]):
        reference = namedtuple("ImageGrid", ("shape", "affine"))(shape=shape, affine=affine)

        # create a nitransforms object
        if fieldmap:
            # compose fieldmap into transform
            raise NotImplementedError
        else:
            initial_xform = Affine(matrix=em_affines[vol_idx], reference=reference)
        mat_file = dirname / f"init_{i_iter}_{vol_idx:05d}.mat"
        initial_xform.to_filename(mat_file, fmt="itk")
        registration.inputs.initial_moving_transform = str(mat_file)

    # execute ants command line
    result = registration.run(cwd=str(dirname)).outputs

    # read output transform
    xform = nt.linear.Affine(
        nt.io.itk.ITKLinearTransform.from_filename(result.forward_transforms[0]).to_ras(
            reference=fixed, moving=moving
        ),
    )
    # debugging: generate aligned file for testing
    xform.apply(moving, reference=fixed).to_filename(
        dirname / f"aligned{vol_idx:05d}_{int(bval):04d}.nii.gz"
    )

    return xform
