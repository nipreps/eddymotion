"""
Vector prediction interfaces
"""
import os
from pathlib import Path
import numpy as np
from nipype.interfaces.base import (
    SimpleInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    isdefined,
)
from dmriprep.utils.vectors import DiffusionGradientTable, B0_THRESHOLD


class _ReorientVectorsInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True)
    rasb_file = File(exists=True)
    affines = traits.List()
    b0_threshold = traits.Float(B0_THRESHOLD, usedefault=True)


class _ReorientVectorsOutputSpec(TraitedSpec):
    out_rasb = File(exists=True)


class ReorientVectors(SimpleInterface):
    """
    Reorient Vectors
    Example
    -------
    >>> os.chdir(tmpdir)
    >>> oldrasb = str(data_dir / 'dwi.tsv')
    >>> oldrasb_mat = np.loadtxt(str(data_dir / 'dwi.tsv'), skiprows=1)
    >>> # The simple case: all affines are identity
    >>> affine_list = np.zeros((len(oldrasb_mat[:, 3][oldrasb_mat[:, 3] != 0]), 4, 4))
    >>> for i in range(4):
    >>>     affine_list[:, i, i] = 1
    >>>     reor_vecs = ReorientVectors()
    >>> reor_vecs = ReorientVectors()
    >>> reor_vecs.inputs.affines = affine_list
    >>> reor_vecs.inputs.in_rasb = oldrasb
    >>> res = reor_vecs.run()
    >>> out_rasb = res.outputs.out_rasb
    >>> out_rasb_mat = np.loadtxt(out_rasb, skiprows=1)
    >>> assert oldrasb_mat == out_rasb_mat
    True
    """

    input_spec = _ReorientVectorsInputSpec
    output_spec = _ReorientVectorsOutputSpec

    def _run_interface(self, runtime):
        from nipype.utils.filemanip import fname_presuffix

        table = DiffusionGradientTable(
            dwi_file=self.inputs.dwi_file,
            rasb_file=self.inputs.rasb_file,
            transforms=self.inputs.affines,
        )
        table.generate_vecval()
        reor_table = table.reorient_rasb()

        cwd = Path(runtime.cwd).absolute()
        reor_rasb_file = fname_presuffix(
            self.inputs.rasb_file,
            use_ext=False,
            suffix="_reoriented.tsv",
            newpath=str(cwd),
        )
        np.savetxt(
            str(reor_rasb_file),
            reor_table,
            delimiter="\t",
            header="\t".join("RASB"),
            fmt=["%.8f"] * 3 + ["%g"],
        )

        self._results["out_rasb"] = reor_rasb_file
        return runtime
