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
"""Parser module."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import yaml


def _parse_yaml_config(file_path: Path) -> dict:
    """
    Parse YAML configuration file.

    Parameters
    ----------
    file_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        A dictionary containing the parsed YAML configuration.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def _build_parser() -> ArgumentParser:
    """
    Build parser object.

    Returns
    -------
    :obj:`~argparse.ArgumentParser`
        The parser object defining the interface for the command-line.
    """
    parser = ArgumentParser(
        description="A model-based algorithm for the realignment of 4D brain images.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_file",
        action="store",
        type=Path,
        help="Path to the HDF5 file containing the original DWI data.",
    )
    parser.add_argument(
        "--align_config",
        action="store",
        type=_parse_yaml_config,
        default=None,
        help="Path to the yaml file containing the parameters to configure the image registration process.",
    )
    parser.add_argument(
        "--models",
        action="store",
        nargs="+",
        default=["b0"],
        help="Select the diffusion model for registration targets.",
    )
    parser.add_argument(
        "--nthreads",
        action="store",
        type=int,
        default=None,
        help="Maximum number of threads an individual process may use.",
    )
    parser.add_argument(
        "--njobs",
        action="store",
        type=int,
        default=None,
        help="Number of parallel jobs.",
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        default=None,
        help="Seed the random number generator for deterministic estimation.",
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        type=Path,
        default=Path.cwd(),
        help="Path to the output directory. Defaults to the current directory. The output file will have the same name as the input file.",
    )

    return parser


def parse_args(args: Optional[list] = None, namespace: Optional[Namespace] = None) -> Namespace:
    """
    Parse args and run further checks on the command line.

    Parameters
    ----------
    args : list of str, optional
        List of strings representing the command line arguments. Defaults to None.
    namespace : :class:`~argparse.Namespace`, optional
        An object to parse the arguments into. Defaults to None.

    Returns
    -------
    :class:`~argparse.Namespace`
        An object holding the parsed arguments.
    """
    parser = _build_parser()
    return parser.parse_args(args, namespace)
