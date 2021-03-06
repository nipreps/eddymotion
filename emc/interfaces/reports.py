"""
Reporting tools
"""
import os
import pandas as pd
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec,
    File, Directory, InputMultiObject, Str, isdefined,
    SimpleInterface)
from emc.utils.viz import _iteration_summary_plot, before_after_images
import matplotlib
matplotlib.use('agg')


class SummaryOutputSpec(TraitedSpec):
    out_report = File(exists=True, desc='HTML segment containing summary')


class SummaryInterface(SimpleInterface):
    output_spec = SummaryOutputSpec

    def _run_interface(self, runtime):
        segment = self._generate_segment()
        fname = os.path.join(runtime.cwd, 'report.html')
        with open(fname, 'w') as fobj:
            fobj.write(segment)
        self._results['out_report'] = fname
        return runtime


class IterationSummaryInputSpec(BaseInterfaceInputSpec):
    collected_motion_files = InputMultiObject(File(exists=True))


class IterationSummaryOutputSpec(TraitedSpec):
    iteration_summary_file = File(exists=True)
    plot_file = File(exists=True)


class IterationSummary(SummaryInterface):
    input_spec = IterationSummaryInputSpec
    output_spec = IterationSummaryOutputSpec

    def _run_interface(self, runtime):
        motion_files = self.inputs.collected_motion_files
        output_fname = os.path.join(runtime.cwd, "iteration_summary.csv")
        fig_output_fname = os.path.join(runtime.cwd, "iterdiffs.svg")
        if not isdefined(motion_files):
            return runtime

        all_iters = []
        for fnum, fname in enumerate(motion_files):
            df = pd.read_csv(fname)
            df['iter_num'] = fnum
            path_parts = fname.split(os.sep)
            itername = '' if 'iter' not in path_parts[-3] else path_parts[-3]
            df['iter_name'] = itername
            all_iters.append(df)
        combined = pd.concat(all_iters, axis=0, ignore_index=True)

        combined.to_csv(output_fname, index=False)
        self._results['iteration_summary_file'] = output_fname

        # Create a figure for the report
        _iteration_summary_plot(combined, fig_output_fname)
        self._results['plot_file'] = fig_output_fname

        return runtime


class EMCReportInputSpec(BaseInterfaceInputSpec):
    iteration_summary = File(exists=True)
    registered_images = InputMultiObject(File(exists=True))
    original_images = InputMultiObject(File(exists=True))
    model_predicted_images = InputMultiObject(File(exists=True))


class EMCReportOutputSpec(SummaryOutputSpec):
    plot_file = File(exists=True)


class EMCReport(SummaryInterface):
    input_spec = EMCReportInputSpec
    output_spec = EMCReportOutputSpec

    def _run_interface(self, runtime):
        import imageio
        images = []
        for imagenum, (orig_file, aligned_file, model_file) in enumerate(zip(
                self.inputs.original_images, self.inputs.registered_images,
                self.inputs.model_predicted_images)):

            images.extend(before_after_images(orig_file, aligned_file, 
                                              model_file, imagenum))

        out_file = os.path.join(runtime.cwd, "emc_reg.gif")
        imageio.mimsave(out_file, images, fps=1)
        self._results['plot_file'] = out_file

        del images
        return runtime
