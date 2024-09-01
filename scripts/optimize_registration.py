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
"""Optimize ANTs' configurations."""

import asyncio
import random
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import nibabel as nb
import nitransforms as nt
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario

from eddymotion.registration import ants as erants
from eddymotion.registration import utils

# When inside ipython / jupyter
# import nest_asyncio
# nest_asyncio.apply()

## Generate config dictionary
configdict = {
    # "convergence_threshold": (1e-5, 1e-6),
    "winsorize_lower_quantile": (0.001, 0.2),
    "winsorize_upper_quantile": (0.98, 0.999),
    "transform_parameters": (0.1, 50.0),
    "smoothing_sigmas": (0.0, 8.0),
    "shrink_factors": (1, 4),
    "radius_or_number_of_bins": (3, 7),
    "sampling_percentage": (0.1, 0.8),
    "metric": ["GC", "CC"],
    "sampling_strategy": ["Random", "Regular"],
}
paramspace = ConfigurationSpace(configdict)


async def ants(cmd, cwd, stdout, stderr, semaphore):
    async with semaphore:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
        )
        returncode = await proc.wait()
        return returncode


DATASET_PATH = Path("/data/datasets/hcph")
DWI_RUN = (
    DATASET_PATH / "sub-001" / "ses-038" / "dwi" / "sub-001_ses-038_acq-highres_dir-PA_dwi.nii.gz"
)

WORKDIR = Path.home() / "tmp" / "eddymotiondev"
WORKDIR.mkdir(parents=True, exist_ok=True)

nii = nb.load(DWI_RUN)

bvecs = np.loadtxt(DWI_RUN.parent / DWI_RUN.name.replace(".nii.gz", ".bvec")).T
bvals = np.loadtxt(DWI_RUN.parent / DWI_RUN.name.replace(".nii.gz", ".bval"))
b0mask = bvals < 50
data_b0s = nii.get_fdata()[..., b0mask]
brainmask_path = WORKDIR / "b0_desc-brain_mask.nii.gz"
brainmask_nii = nb.load(brainmask_path)
brainmask = np.asanyarray(brainmask_nii.dataobj) > 1e-5

ribbon_nii = nb.load(WORKDIR / "ribbon.nii.gz")

fixed_path = WORKDIR / "first-b0.nii.gz"
refnii = nb.four_to_three(nii)[0]
refnii.to_filename(fixed_path)

EXPERIMENTDIR = WORKDIR / "smac"
if EXPERIMENTDIR.exists():
    rmtree(EXPERIMENTDIR, ignore_errors=True)

EXPERIMENTDIR.mkdir(parents=True, exist_ok=True)


async def train_coro(config: Configuration, conversions: int = 50, seed: int = 0) -> float:
    random.seed(seed)

    tmp_folder = Path(mkdtemp(prefix="i-", dir=EXPERIMENTDIR))

    ref_xfms = []
    tasks = []
    semaphore = asyncio.Semaphore(12)
    for i in range(conversions):
        r_x, r_y, r_z = (
            random.uniform(-0.1, 0.1),
            random.uniform(-0.1, 0.1),
            random.uniform(-0.1, 0.1),
        )
        t_x, t_y, t_z = (
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
        )
        T = nb.affines.from_matvec(
            nb.eulerangles.euler2mat(x=r_x, y=r_y, z=r_z),
            (t_x, t_y, t_z),
        )
        xfm = nt.linear.Affine(T, reference=refnii)
        ref_xfms.append(xfm)

        moving_path = tmp_folder / f"test-{i:02d}.nii.gz"
        (~xfm).apply(refnii, reference=refnii).to_filename(moving_path)

        align_kwargs = {k: config[k] for k in configdict.keys()}

        cmdline = erants.generate_command(
            fixed_path,
            moving_path,
            fixedmask_path=brainmask_path,
            output_transform_prefix=f"conversion-{i:02d}",
            **align_kwargs,
        )

        tasks.append(
            ants(
                cmdline,
                cwd=str(tmp_folder),
                stdout=Path(tmp_folder / f"ants-{i:04d}.out").open("w+"),
                stderr=Path(tmp_folder / f"ants-{i:04d}.err").open("w+"),
                semaphore=semaphore,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    diff = []
    for i, r in enumerate(results):
        if r:
            return 1e6

        fixednii = nb.load(fixed_path)
        movingnii = nb.load(tmp_folder / f"test-{i:02d}.nii.gz")
        xform = nt.linear.Affine(
            nt.io.itk.ITKLinearTransform.from_filename(
                tmp_folder / f"conversion-{i:02d}0GenericAffine.mat"
            ).to_ras(
                reference=fixednii,
                moving=movingnii,
            ),
        )
        disps = utils.displacements_within_mask(
            ribbon_nii,
            xform,
            ref_xfms[i],
        )
        diff.append(
            0.20 * np.percentile(disps, 95)
            + 0.8 * np.median(disps)
            - 0.2 * np.percentile(disps, 5),
        )

    error = np.mean(diff)

    with Path(tmp_folder / f"ants-{i:04d}.out").open("a") as f:
        f.write(f"Iteration error={error}\n\n")
        f.write(f"Tested parameters:\n{config}")

    return error


def train(config: Configuration, conversions: int = 50, seed: int = 0) -> float:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(train_coro(config, conversions, seed))


# Scenario object specifying the optimization environment
scenario = Scenario(paramspace, n_trials=200)

# Use SMAC to find the best configuration/hyperparameters
smac = HyperparameterOptimizationFacade(scenario, train)
incumbent = smac.optimize()

print(incumbent)
