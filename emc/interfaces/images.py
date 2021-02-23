"""Image tools interfaces."""
import os
import numpy as np
import nibabel as nb
from nipype import logging
from nipype.interfaces import ants
from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import (
    traits,
    TraitedSpec,
    BaseInterfaceInputSpec,
    SimpleInterface,
    File,
    isdefined,
    InputMultiObject,
    OutputMultiObject,
    CommandLine,
)
from emc.utils.images import match_transforms


LOGGER = logging.getLogger("nipype.interface")


class MatchTransformsInputSpec(BaseInterfaceInputSpec):
    b0_indices = traits.List(mandatory=True)
    dwi_files = InputMultiObject(File(exists=True), mandatory=True)
    transforms = InputMultiObject(File(exists=True), mandatory=True)


class MatchTransformsOutputSpec(TraitedSpec):
    transforms = OutputMultiObject(File(exists=True), mandatory=True)


class MatchTransforms(SimpleInterface):
    input_spec = MatchTransformsInputSpec
    output_spec = MatchTransformsOutputSpec

    def _run_interface(self, runtime):
        self._results["transforms"] = match_transforms(
            self.inputs.dwi_files, self.inputs.transforms,
            self.inputs.b0_indices
        )
        return runtime


class N3BiasFieldCorrection(ants.N4BiasFieldCorrection):
    _cmd = "N3BiasFieldCorrection"


class ImageMathInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, position=3, argstr="%s")
    dimension = traits.Enum(3, 2, 4, usedefault=True, argstr="%d", position=0)
    out_file = File(argstr="%s", genfile=True, position=1)
    operation = traits.Str(argstr="%s", position=2)
    secondary_arg = traits.Str("", argstr="%s")
    secondary_file = File(argstr="%s")


class ImageMathOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class ImageMath(CommandLine):
    input_spec = ImageMathInputSpec
    output_spec = ImageMathOutputSpec
    _cmd = "ImageMath"

    def _gen_filename(self, name):
        if name == "out_file":
            output = self.inputs.out_file
            if not isdefined(output):
                _, fname, ext = split_filename(self.inputs.in_file)
                output = fname + "_" + self.inputs.operation + ext
            return output
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = os.path.abspath(self._gen_filename("out_file"))
        return outputs


class CombineMotionsInputSpec(BaseInterfaceInputSpec):
    transform_files = InputMultiObject(
        File(exists=True), mandatory=True, desc="transform files from emc"
    )
    source_files = InputMultiObject(
        File(exists=True), mandatory=True, desc="Moving images"
    )
    ref_file = File(exists=True, mandatory=True, desc="Fixed Image")


class CombineMotionsOututSpec(TraitedSpec):
    motion_file = File(exists=True)


class CombineMotions(SimpleInterface):
    input_spec = CombineMotionsInputSpec
    output_spec = CombineMotionsOututSpec

    def _run_interface(self, runtime):
        import pandas as pd
        from emc.utils.images import get_params

        output_fname = os.path.join(runtime.cwd, "motion_params.csv")
        motion_parms_path = os.path.join(runtime.cwd, "movpar.txt")
        motion_params = open(os.path.abspath(motion_parms_path), "w")

        collected_motion = []
        for aff in self.inputs.transform_files:
            rotations, translations = get_params(np.load(aff))
            collected_motion.append(rotations + translations)
            for i in rotations + translations:
                motion_params.write("%f " % i)
            motion_params.write("\n")
        motion_params.close()

        final_motion = np.row_stack(collected_motion)
        cols = ["rotateX", "rotateY", "rotateZ", "shiftX", "shiftY", "shiftZ"]
        motion_df = pd.DataFrame(data=final_motion, columns=cols)
        motion_df.to_csv(output_fname, index=False)
        self._results['motion_file'] = output_fname

        return runtime


class CalculateCNRInputSpec(BaseInterfaceInputSpec):
    emc_warped_images = InputMultiObject(File(exists=True))
    predicted_images = InputMultiObject(File(exists=True))
    mask_image = File(exists=True)


class CalculateCNROutputSpec(TraitedSpec):
    cnr_image = File(exists=True)


class CalculateCNR(SimpleInterface):
    input_spec = CalculateCNRInputSpec
    output_spec = CalculateCNROutputSpec

    def _run_interface(self, runtime):
        from emc.utils.images import rapid_load

        cnr_file = os.path.join(runtime.cwd, "emc_CNR.nii.gz")
        model_images = rapid_load(self.inputs.predicted_images)
        observed_images = rapid_load(self.inputs.emc_warped_images)
        mask_image = nb.load(self.inputs.mask_image)
        mask = mask_image.get_data() > 1e-6
        signal_vals = model_images[mask]
        b0 = signal_vals[:, 0][:, np.newaxis]
        signal_vals = signal_vals / b0
        signal_var = np.var(signal_vals, 1)
        observed_vals = observed_images[mask] / b0
        noise_var = np.var(signal_vals - observed_vals, 1)
        snr = np.nan_to_num(signal_var / noise_var)
        out_mat = np.zeros(mask_image.shape)
        out_mat[mask] = snr
        nb.Nifti1Image(
            out_mat, mask_image.affine, header=mask_image.header
        ).to_filename(cnr_file)
        self._results["cnr_image"] = cnr_file
        return runtime


class ReorderOutputsInputSpec(BaseInterfaceInputSpec):
    b0_indices = traits.List(mandatory=True)
    b0_median = File(exists=True, mandatory=True)
    warped_b0_images = InputMultiObject(File(exists=True), mandatory=True)
    warped_dwi_images = InputMultiObject(File(exists=True), mandatory=True)
    initial_transforms = InputMultiObject(File(exists=True), mandatory=True)
    model_based_transforms = InputMultiObject(traits.List(), mandatory=True)
    model_predicted_images = InputMultiObject(File(exists=True), mandatory=True)


class ReorderOutputsOutputSpec(TraitedSpec):
    full_transforms = OutputMultiObject(traits.List())
    full_predicted_dwi_series = OutputMultiObject(File(exists=True))
    emc_warped_images = OutputMultiObject(File(exists=True))


class ReorderOutputs(SimpleInterface):
    input_spec = ReorderOutputsInputSpec
    output_spec = ReorderOutputsOutputSpec

    def _run_interface(self, runtime):
        full_transforms = []
        full_predicted_dwi_series = []
        full_warped_images = []
        warped_b0_images = self.inputs.warped_b0_images[::-1]
        warped_dwi_images = self.inputs.warped_dwi_images[::-1]
        model_transforms = self.inputs.model_based_transforms[::-1]
        model_images = self.inputs.model_predicted_images[::-1]
        b0_transforms = [
            self.inputs.initial_transforms[idx] for idx in self.inputs.b0_indices
        ][::-1]
        num_dwis = len(self.inputs.initial_transforms)

        for imagenum in range(num_dwis):
            if imagenum in self.inputs.b0_indices:
                full_predicted_dwi_series.append(self.inputs.b0_median)
                full_transforms.append(b0_transforms.pop())
                full_warped_images.append(warped_b0_images.pop())
            else:
                full_transforms.append(model_transforms.pop())
                full_predicted_dwi_series.append(model_images.pop())
                full_warped_images.append(warped_dwi_images.pop())

        if not len(model_transforms) == len(b0_transforms) == len(model_images) == 0:
            raise Exception("Unable to recombine images and transforms")

        self._results["emc_warped_images"] = full_warped_images
        self._results["full_transforms"] = full_transforms
        self._results["full_predicted_dwi_series"] = full_predicted_dwi_series

        return runtime
