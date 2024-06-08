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

import sys

import pytest

from eddymotion.__main__ import main


@pytest.fixture(autouse=True)
def set_command(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["eddymotion"])
        yield


def test_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert captured.out.startswith("usage: eddymotion [-h]")


def test_main(tmp_path, datadir):
    input_file = datadir / "dwi.h5"

    with pytest.raises(SystemExit):
        main([str(input_file), str(tmp_path)])

    with pytest.raises(SystemExit):
        main(
            [
                str(input_file),
                "--models",
                "b0",
                "--omp_nthreads",
                "1",
                "--n_jobs",
                "1",
                "--seed",
                "1234",
                "--output_dir",
                str(tmp_path),
            ]
        )
    # assert Path(output_dir).joinpath("dwi.h5").exists()  # Empty

    # Also, call python -m eddymotion or eddymotion from CircleCI ??
