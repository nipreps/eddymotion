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

"""Test parser."""

import pytest

from eddymotion.cli.parser import _build_parser

MIN_ARGS = ["data/dwi.h5"]


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ([], 2),
    ],
)
def test_parser_errors(args, code):
    """Check behavior of the parser."""
    with pytest.raises(SystemExit) as error:
        _build_parser().parse_args(args)

    assert error.value.code == code


@pytest.mark.parametrize(
    "args",
    [
        MIN_ARGS,
    ],
)
def test_parser_valid(tmp_path, args):
    """Check valid arguments."""
    datapath = tmp_path / "data"
    datapath.mkdir(exist_ok=True)
    args[0] = str(datapath)

    opts = _build_parser().parse_args(args)

    assert opts.input_file == datapath
    assert opts.models == ["b0"]


@pytest.mark.parametrize(
    ("argval", "_models"),
    [
        ("b0", "b0"),
        ("s0", "s0"),
        ("avg", "avg"),
        ("average", "average"),
        ("mean", "mean"),
    ],
)
def test_models_arg(tmp_path, argval, _models):
    """Check the correct parsing of the models argument."""
    datapath = tmp_path / "data"
    datapath.mkdir(exist_ok=True)

    args = [str(datapath)] + ["--models", argval]
    opts = _build_parser().parse_args(args)

    assert opts.models == [_models]
