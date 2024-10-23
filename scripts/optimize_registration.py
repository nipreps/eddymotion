# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
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
import logging
from os import getenv
from pathlib import Path
from shutil import rmtree

import nibabel as nb
import nitransforms as nt
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario
from smac.utils.configspace import get_config_hash

from eddymotion.registration import ants as erants
from eddymotion.registration import utils

logger = logging.getLogger("ants-optimization")

# When inside ipython / jupyter
# import nest_asyncio
# nest_asyncio.apply()

TIME_PENALTY_WEIGHT = 0.1
SEED = 2390232

## Generate config dictionary
configdict = {
    # "convergence_threshold": (1e-5, 1e-6),
    # "winsorize_lower_quantile": (0.001, 0.2),
    # "winsorize_upper_quantile": (0.9, 0.999),
    "transform_parameters": (0.01, 2.0),
    "smoothing_sigmas": (0.0, 1.0),
    "shrink_factors": (1, 2),
    "radius_or_number_of_bins": (3, 5),
    "sampling_percentage": (0.1, 0.4),
    # "metric": ["GC"],
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


DATASET_PATH = Path(getenv("TEST_DATA_HOME", f"{getenv('HOME')}/.cache/eddymotion-tests"))

WORKDIR = Path.home() / "tmp" / "eddymotiondev"
WORKDIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTDIR = WORKDIR / "smac"
if EXPERIMENTDIR.exists():
    rmtree(EXPERIMENTDIR, ignore_errors=True)

EXPERIMENTDIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)
MOTION_PARAMETERS = np.hstack(
    (rng.uniform(-0.4, 0.4, size=(60, 3)), rng.uniform(-2.0, 2.0, size=(60, 3)))
)
CONVERSIONS = [
    nb.affines.from_matvec(nb.eulerangles.euler2mat(*parameters[:3]), parameters[3:])
    for parameters in MOTION_PARAMETERS
]

REFERENCES = (
    DATASET_PATH / "dwi-b0_desc-avg.nii.gz",
    DATASET_PATH / "hcph-b0_desc-avg.nii.gz",
)


async def train_coro(
    config: Configuration,
    seed: int = 0,
    verbose: bool = False,
) -> float:
    tmp_folder = EXPERIMENTDIR / get_config_hash(config)
    tmp_folder.mkdir(parents=True, exist_ok=True)
    align_kwargs = {k: config[k] for k in configdict.keys()}

    ref_xfms = []
    tasks = []
    semaphore = asyncio.Semaphore(18)
    nconv = len(CONVERSIONS)
    for i, T in enumerate(CONVERSIONS):
        for j in (0, 1):
            fixed_path = REFERENCES[j]
            brainmask_path = DATASET_PATH / fixed_path.name.replace("desc-avg", "desc-brain_mask")
            refnii = nb.load(fixed_path)
            xfm = nt.linear.Affine(T, reference=refnii)
            ref_xfms.append(xfm)

            index = i * len(REFERENCES) + j
            moving_path = tmp_folder / f"test-{index:04d}.nii.gz"
            (~xfm).apply(refnii, reference=refnii).to_filename(moving_path)

            cmdline = erants.generate_command(
                fixed_path,
                moving_path,
                fixedmask_path=brainmask_path,
                output_transform_prefix=f"conversion-{index:04d}",
                **align_kwargs,
            )

            tasks.append(
                ants(
                    cmdline,
                    cwd=str(tmp_folder),
                    stdout=Path(tmp_folder / f"ants-{index:04d}.out").open("w+"),
                    stderr=Path(tmp_folder / f"ants-{index:04d}.err").open("w+"),
                    semaphore=semaphore,
                )
            )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    diff = []
    times = []
    start = []
    for i, r in enumerate(results):
        if r:
            return 1e6

        j = i % 2
        fixed_path = REFERENCES[j]
        brainmask_path = DATASET_PATH / fixed_path.name.replace("desc-avg", "desc-brain_mask")

        fixednii = nb.load(fixed_path)
        movingnii = nb.load(tmp_folder / f"test-{i:04d}.nii.gz")
        xform = nt.linear.Affine(
            nt.io.itk.ITKLinearTransform.from_filename(
                tmp_folder / f"conversion-{i:04d}0GenericAffine.mat"
            ).to_ras(
                reference=fixednii,
                moving=movingnii,
            ),
        )

        masknii = nb.load(brainmask_path)
        initial_error = utils.displacements_within_mask(
            masknii,
            ref_xfms[i],
        )

        disps = utils.displacements_within_mask(
            masknii,
            xform,
            ref_xfms[i],
        )
        diff.append(np.percentile(disps, 95))
        start.append(np.percentile(initial_error, 95))

        # Parse log -- Total elapsed time: 1.0047e+00
        for line in reversed(Path(tmp_folder / f"ants-{i:04d}.out").read_text().splitlines()):
            if line.strip().startswith("Total elapsed time:"):
                times.append(float(line.strip().split(" ")[-1]))

    meandiff = np.mean(diff)
    meantime = np.mean(times)
    error = ((1.0 - TIME_PENALTY_WEIGHT) * meandiff + TIME_PENALTY_WEIGHT * meantime) / np.mean(
        start
    )

    logger.info(
        f"Normalized objective ({nconv} it.): {error:0.3f} "
        f"({meandiff:0.2f} mm | {meantime:0.2f} s). "
        f"Avg. p95 initial error: {np.mean(start):0.2f} mm."
    )
    if verbose:
        logger.info(f"\n\nParameters:\n{align_kwargs}" f"\n\nConversions folder: {tmp_folder}.")

    return error


def train(config: Configuration, seed: int = 0) -> float:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(train_coro(config, seed))


# Scenario object specifying the optimization environment
scenario = Scenario(paramspace, n_trials=200)

# Use SMAC to find the best configuration/hyperparameters
smac = HyperparameterOptimizationFacade(scenario, train)
incumbent = smac.optimize()

print(incumbent)

loop = asyncio.get_event_loop()
loop.run_until_complete(train_coro(incumbent, verbose=True))
