# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The NiPreps Developers <nipreps@gmail.com>
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
"""Using ANTs for image registration."""

from collections import namedtuple
from json import loads
from pathlib import Path
from warnings import warn

import nibabel as nb
import nitransforms as nt
import numpy as np
from nipype.interfaces.ants.registration import Registration
from nitransforms.linear import Affine
from pkg_resources import resource_filename as pkg_fn

PARAMETERS_SINGLE_VALUE = {
    "collapse_output_transforms",
    "dimension",
    "initial_moving_transform",
    "initialize_transforms_per_stage",
    "interpolation",
    "output_transform_prefix",
    "verbose",
    "winsorize_lower_quantile",
    "winsorize_upper_quantile",
    "write_composite_transform",
}

PARAMETERS_SINGLE_LIST = {
    "radius_or_number_of_bins",
    "sampling_percentage",
    "metric",
    "sampling_strategy",
}
PARAMETERS_DOUBLE_LIST = {"shrink_factors", "smoothing_sigmas", "transform_parameters"}


def _to_nifti(data, affine, filename, clip=True):
    data = np.squeeze(data)
    if clip:
        from eddymotion.data.filtering import advanced_clip

        data = advanced_clip(data)
    nii = nb.Nifti1Image(
        data,
        affine,
        None,
    )
    nii.header.set_sform(affine, code=1)
    nii.header.set_qform(affine, code=1)
    nii.to_filename(filename)


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


def _get_ants_settings(settings="b0-to-b0_level0"):
    return Path(
        pkg_fn(
            "eddymotion.registration",
            f"config/{settings}.json",
        )
    )


def _massage_mask_path(mask_path, nlevels):
    """
    Generate nipype-compatible masks paths.

    Examples
    --------
    >>> _massage_mask_path("/some/path", 2)
    ['/some/path', '/some/path']

    >>> _massage_mask_path(["/some/path"] * 2, 2)
    ['/some/path', '/some/path']

    >>> _massage_mask_path(["/some/path"] * 2, 4)
    ['NULL', 'NULL', '/some/path', '/some/path']

    >>> _massage_mask_path(["/some/path"] * 2, 1)
    ['/some/path']

    """
    if isinstance(mask_path, (str, Path)):
        return [str(mask_path)] * nlevels
    if len(mask_path) < nlevels:
        return ["NULL"] * (nlevels - len(mask_path)) + mask_path
    if len(mask_path) > nlevels:
        warn("More mask paths than levels", stacklevel=1)
        return mask_path[:nlevels]
    return mask_path


def generate_command(
    fixed_path,
    moving_path,
    fixedmask_path=None,
    movingmask_path=None,
    init_affine=None,
    default="b0-to-b0_level0",
    **kwargs,
):
    """
    Generate an ANTs' command line.

    Examples
    --------
    >>> generate_command(
    ...     fixed_path=repodata / 'fileA.nii.gz',
    ...     moving_path=repodata / 'fileB.nii.gz',
    ... )  # doctest: +NORMALIZE_WHITESPACE
    'antsRegistration --collapse-output-transforms 1 --dimensionality 3 \
    --initialize-transforms-per-stage 0 --interpolation Linear --output transform \
    --transform Rigid[ 12.0 ] \
    --metric GC[ .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 3, Random, 0.4 ] \
    --convergence [ 20, 1e-06, 4 ] --smoothing-sigmas 2.71vox --shrink-factors 3 \
    --use-histogram-matching 1 \
    --transform Rigid[ 1.96 ] \
    --metric GC[ .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 4, Random, 0.18 ] \
    --convergence [ 10, 1e-07, 2 ] --smoothing-sigmas 0.0vox --shrink-factors 2 \
    --use-histogram-matching 1 \
    -v --winsorize-image-intensities [ 0.063, 0.991 ] \
    --write-composite-transform 0'

    >>> generate_command(
    ...     fixed_path=repodata / 'fileA.nii.gz',
    ...     moving_path=repodata / 'fileB.nii.gz',
    ...     default="dwi-to-b0_level0",
    ... )  # doctest: +NORMALIZE_WHITESPACE
    'antsRegistration --collapse-output-transforms 1 --dimensionality 3 \
    --initialize-transforms-per-stage 0 --interpolation Linear --output transform \
    --transform Rigid[ 0.01 ] --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Regular, 0.2 \
    ] --convergence [ 100x50, 1e-05, 10 ] --smoothing-sigmas 2.0x0.0vox \
    --shrink-factors 2x1 --use-histogram-matching 1 --transform Rigid[ 0.001 ] \
    --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Random, 0.1 \
    ] --convergence [ 25, 1e-06, 2 ] --smoothing-sigmas 0.0vox --shrink-factors 1 \
    --use-histogram-matching 1 -v --winsorize-image-intensities [ 0.0001, 0.9998 ] \
    --write-composite-transform 0'

    >>> generate_command(
    ...     fixed_path=repodata / 'fileA.nii.gz',
    ...     moving_path=repodata / 'fileB.nii.gz',
    ...     fixedmask_path=repodata / 'maskA.nii.gz',
    ...     default="dwi-to-b0_level0",
    ... )  # doctest: +NORMALIZE_WHITESPACE
    'antsRegistration --collapse-output-transforms 1 --dimensionality 3 \
    --initialize-transforms-per-stage 0 --interpolation Linear --output transform \
    --transform Rigid[ 0.01 ] --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Regular, 0.2 ] \
    --convergence [ 100x50, 1e-05, 10 ] --smoothing-sigmas 2.0x0.0vox --shrink-factors 2x1 \
    --use-histogram-matching 1 --masks [ \
        .../maskA.nii.gz, NULL ] \
    --transform Rigid[ 0.001 ] --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Random, 0.1 ] \
    --convergence [ 25, 1e-06, 2 ] --smoothing-sigmas 0.0vox --shrink-factors 1 \
    --use-histogram-matching 1 --masks [ \
        .../maskA.nii.gz, NULL ] \
    -v --winsorize-image-intensities [ 0.0001, 0.9998 ]  --write-composite-transform 0'

    >>> generate_command(
    ...     fixed_path=repodata / 'fileA.nii.gz',
    ...     moving_path=repodata / 'fileB.nii.gz',
    ...     default="dwi-to-b0_level0",
    ... )  # doctest: +NORMALIZE_WHITESPACE
    'antsRegistration --collapse-output-transforms 1 --dimensionality 3 \
    --initialize-transforms-per-stage 0 --interpolation Linear --output transform \
    --transform Rigid[ 0.01 ] --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Regular, 0.2 \
    ] --convergence [ 100x50, 1e-05, 10 ] --smoothing-sigmas 2.0x0.0vox \
    --shrink-factors 2x1 --use-histogram-matching 1 --transform Rigid[ 0.001 ] \
    --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Random, 0.1 \
    ] --convergence [ 25, 1e-06, 2 ] --smoothing-sigmas 0.0vox --shrink-factors 1 \
    --use-histogram-matching 1 -v --winsorize-image-intensities [ 0.0001, 0.9998 ] \
    --write-composite-transform 0'

    >>> generate_command(
    ...     fixed_path=repodata / 'fileA.nii.gz',
    ...     moving_path=repodata / 'fileB.nii.gz',
    ...     fixedmask_path=[repodata / 'maskA.nii.gz'],
    ...     default="dwi-to-b0_level0",
    ... )  # doctest: +NORMALIZE_WHITESPACE
    'antsRegistration --collapse-output-transforms 1 --dimensionality 3 \
    --initialize-transforms-per-stage 0 --interpolation Linear --output transform \
    --transform Rigid[ 0.01 ] --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Regular, 0.2 ] \
    --convergence [ 100x50, 1e-05, 10 ] --smoothing-sigmas 2.0x0.0vox --shrink-factors 2x1 \
    --use-histogram-matching 1 --masks [ NULL, NULL ] \
    --transform Rigid[ 0.001 ] --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Random, 0.1 ] \
    --convergence [ 25, 1e-06, 2 ] --smoothing-sigmas 0.0vox --shrink-factors 1 \
    --use-histogram-matching 1 --masks [ \
        .../maskA.nii.gz, NULL ] \
    -v --winsorize-image-intensities [ 0.0001, 0.9998 ]  --write-composite-transform 0'

    """

    # Bootstrap settings from defaults file and override with single-valued parameters in args
    settings = loads(_get_ants_settings(default).read_text()) | {
        k: kwargs.pop(k) for k in PARAMETERS_SINGLE_VALUE if k in kwargs
    }

    # Determine number of levels and assert consistency of levels
    levels = {len(settings[p]) for p in PARAMETERS_SINGLE_LIST if p in settings}
    nlevels = levels.pop()
    if levels:
        raise RuntimeError(f"Malformed settings file (levels: {levels})")

    # Override list (and nested-list) parameters
    for key, value in kwargs.items():
        if key in PARAMETERS_DOUBLE_LIST:
            value = [value]
        elif key not in PARAMETERS_SINGLE_LIST:
            continue

        if levels == 1:
            settings[key] = [value]
        else:
            settings[key][-1] = value

    # Set fixed masks if provided
    if fixedmask_path is not None:
        settings["fixed_image_masks"] = [
            str(p) for p in _massage_mask_path(fixedmask_path, nlevels)
        ]

    # Set moving masks if provided
    if movingmask_path is not None:
        settings["moving_image_masks"] = [
            str(p) for p in _massage_mask_path(movingmask_path, nlevels)
        ]

    # Set initalizing affine if provided
    if init_affine is not None:
        settings["initial_moving_transform"] = str(init_affine)

    # Generate command line with nipype and return
    return Registration(
        fixed_image=str(Path(fixed_path).absolute()),
        moving_image=str(Path(moving_path).absolute()),
        **settings,
    ).cmdline


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

    if isinstance(reg_target_type, str):
        reg_target_type = (reg_target_type, reg_target_type)

    registration = Registration(
        terminal_output="file",
        from_file=pkg_fn(
            "eddymotion.registration",
            f"config/{reg_target_type[0]}-to-{reg_target_type[1]}_level{i_iter}.json",
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
