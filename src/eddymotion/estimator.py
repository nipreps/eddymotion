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

from pathlib import Path
from tempfile import TemporaryDirectory, mkstemp

import nibabel as nb
from tqdm import tqdm

from eddymotion import utils as eutils
from eddymotion.data.splitting import lovo_split
from eddymotion.model.base import ModelFactory
from eddymotion.registration.ants import _prepare_registration_data, _run_registration


class EddyMotionEstimator:
    """Estimates rigid-body head-motion and distortions derived from eddy-currents."""

    @staticmethod
    def estimate(
        data,
        *,
        align_kwargs=None,
        iter_kwargs=None,
        models=("b0",),
        omp_nthreads=None,
        n_jobs=None,
        **kwargs,
    ):
        r"""
        Estimate head-motion and Eddy currents.

        Parameters
        ----------
        data : :obj:`~eddymotion.dmri.DWI`
            The target DWI dataset, represented by this tool's internal
            type. The object is used in-place, and will contain the estimated
            parameters in its ``em_affines`` property, as well as the rotated
            *b*-vectors within its ``gradients`` property.
        n_iter : :obj:`int`
            Number of iterations this particular model is going to be repeated.
        align_kwargs : :obj:`dict`
            Parameters to configure the image registration process.
        iter_kwargs : :obj:`dict`
            Parameters to configure the iterator strategy to traverse timepoints/orientations.
        models : :obj:`list`
            Selects the diffusion model that will generate the registration target
            corresponding to each gradient map.
            See :obj:`~eddymotion.model.ModelFactory` for allowed models (and corresponding
            keywords).
        omp_nthreads : :obj:`int`
            Maximum number of threads an individual process may use.
        n_jobs : :obj:`int`
            Number of parallel jobs.

        Return
        ------
        :obj:`list` of :obj:`numpy.ndarray`
            A list of :math:`4 \times 4` affine matrices encoding the estimated
            parameters of the deformations caused by head-motion and eddy-currents.

        """

        # Massage iterator configuration
        iter_kwargs = iter_kwargs or {}
        iter_kwargs = {
            "seed": None,
            "bvals": None,  # TODO: extract b-vals here if pertinent
        } | iter_kwargs
        iter_kwargs["size"] = len(data)

        iterfunc = getattr(eutils, f'{iter_kwargs.pop("strategy", "random")}_iterator')
        index_order = list(iterfunc(**iter_kwargs))

        align_kwargs = align_kwargs or {}

        if "num_threads" not in align_kwargs and omp_nthreads is not None:
            align_kwargs["num_threads"] = omp_nthreads

        n_iter = len(models)

        reg_target_type = (
            align_kwargs.pop("fixed_modality", None),
            align_kwargs.pop("moving_modality", None),
        )

        for i_iter, model in enumerate(models):
            # When downsampling these need to be set per-level
            bmask_img = _prepare_brainmask_data(data.brainmask, data.affine)

            _prepare_kwargs(data, kwargs)

            single_model = model.lower() in (
                "b0",
                "s0",
                "avg",
                "average",
                "mean",
                "gp",
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
                dwmodel.fit(data.dataobj, n_jobs=n_jobs)

            with TemporaryDirectory() as tmp_dir:
                print(f"Processing in <{tmp_dir}>")
                ptmp_dir = Path(tmp_dir)
                with tqdm(total=len(index_order), unit="dwi") as pbar:
                    # run a original-to-synthetic affine registration
                    for i in index_order:
                        pbar.set_description_str(
                            f"Pass {i_iter + 1}/{n_iter} | Fit and predict b-index <{i}>"
                        )
                        data_train, data_test = lovo_split(data, i, with_b0=True)
                        grad_str = f"{i}, {data_test[1][:3]}, b={int(data_test[1][3])}"
                        pbar.set_description_str(f"[{grad_str}], {n_jobs} jobs")

                        if not single_model:  # A true LOGO estimator
                            if hasattr(data, "gradients"):
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
                            data_test[0], predicted, data.affine, i, ptmp_dir, reg_target_type
                        )

                        pbar.set_description_str(
                            f"Pass {i_iter + 1}/{n_iter} | Realign b-index <{i}>"
                        )

                        xform = _run_registration(
                            fixed,
                            moving,
                            bmask_img,
                            data.em_affines,
                            data.affine,
                            data.dataobj.shape[:3],
                            data_test[1][3],
                            data.fieldmap,
                            i_iter,
                            i,
                            ptmp_dir,
                            reg_target_type,
                            align_kwargs,
                        )

                        # update
                        data.set_transform(i, xform.matrix)
                        pbar.update()

        return data.em_affines


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


def _prepare_kwargs(data, kwargs):
    """Prepare the keyword arguments depending on the DWI data: add attributes corresponding to
    the ``brainmask``, ``bzero``, ``gradients``, ``frame_time``, and ``total_duration`` DWI data
    properties.

    Modifies kwargs in-place.

    Parameters
    ----------
    data : :class:`eddymotion.data.dmri.DWI`
        DWI data object.
    kwargs : :obj:`dict`
        Keyword arguments.
    """
    from eddymotion.data.filtering import advanced_clip as _advanced_clip

    if data.brainmask is not None:
        kwargs["mask"] = data.brainmask

    if hasattr(data, "bzero") and data.bzero is not None:
        kwargs["S0"] = _advanced_clip(data.bzero)

    if hasattr(data, "gradients"):
        kwargs["gtab"] = data.gradients

    if hasattr(data, "frame_time"):
        kwargs["timepoints"] = data.frame_time

    if hasattr(data, "total_duration"):
        kwargs["xlim"] = data.total_duration
