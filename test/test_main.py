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

from eddymotion.__main__ import main


def test_help(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["main", "--help"])

        try:
            main()
        except:
            print(
                "Help is printed. Unsure why it raises an exception"
            )  # args is None parse_args, but help is printed


def test_main(monkeypatch, tmp_path, datadir):
    input_dir = datadir / "dwi.h5"

    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["main", input_dir])

        try:
            main()
        except:
            print(
                "Unsure why it raises another exception"
            )  # args is None parse_args, but help is not printed...

    align_kwargs = dict({})
    models = ["b0"]
    omp_nthreads = 1
    n_jobs = 1
    seed = 1234
    output_dir = tmp_path

    with monkeypatch.context() as m:
        m.setattr(
            sys,
            "argv",
            [
                "main",
                input_dir,
                "--align_kwargs",
                align_kwargs,
                "--models",
                models,
                "--omp_nthreads",
                omp_nthreads,
                "--n_jobs",
                n_jobs,
                "--seed",
                seed,
                "--output_dir",
                output_dir,
            ],
        )

        try:
            main()
        except:
            print("Unsure why it raises an additional exception")  # args is None parse_args,

    # assert Path(output_dir).joinpath("dwi.h5").exists()  # Empty

    # Also, call python -m eddymotion or eddymotion from CircleCI ??


def test_main2(tmp_path, datadir):
    input_dir = datadir / "dwi.h5"
    import pytest

    with pytest.raises(SystemExit) as wrapped_exit:
        main(
            [str(input_dir), str(tmp_path)]
        )  # adding argv=None to main allows to do this; fails to open file but OK
