"""Prediction interfaces."""
from pathlib import Path
import nibabel as nb
import numpy as np
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, SimpleInterface, File,
    Directory, Str, isdefined, InputMultiObject, OutputMultiPath,
    OutputMultiObject, CommandLine, CommandLineInputSpec
)


class SignalPredictionInputSpec(BaseInterfaceInputSpec):
    aligned_dwi_files = InputMultiObject(File(exists=True), mandatory=True)
    aligned_vectors = File(exists=True, mandatory=True)
    b0_mask = File(exists=True, mandatory=True)
    b0_median = File(exists=True, mandatory=True)
    bvec_to_predict = traits.Array()
    bval_to_predict = traits.Float()
    minimal_q_distance = traits.Float(2.0, usedefault=True)
    b0_indices = traits.List()
    prune_b0s = traits.Bool(False, usedefault=True)
    model_name = traits.Str(default_value="sfm", userdefault=True)


class SignalPredictionOutputSpec(TraitedSpec):
    predicted_image = File(exists=True)


class SignalPrediction(SimpleInterface):
    """
    Predicts signal for each dwi volume from a list of 3D coordinate vectors
    along the sphere.
    """

    input_spec = SignalPredictionInputSpec
    output_spec = SignalPredictionOutputSpec

    def _run_interface(self, runtime):
        import warnings
        import time
        from emc.utils.images import series_files2series_arr
        from emc.utils.vectors import _nonoverlapping_qspace_samples
        from dipy.core.gradients import gradient_table_from_bvals_bvecs
        from dipy.reconst.shore import ShoreModel
        from dipy.reconst.dti import TensorModel
        from emc.utils.model import SFM4HMC, ExponentialIsotropicModel
        from emc.utils.images import prune_b0s_from_dwis
        warnings.filterwarnings("ignore")

        pred_vec = self.inputs.bvec_to_predict
        pred_val = self.inputs.bval_to_predict

        # Load the mask image:
        mask_img = nb.load(self.inputs.b0_mask)
        mask_array = mask_img.get_fdata() > 1e-6

        if self.inputs.prune_b0s is True:
            all_images = prune_b0s_from_dwis(
                self.inputs.aligned_dwi_files, self.inputs.b0_indices
            )
        else:
            all_images = self.inputs.aligned_dwi_files

        # Load the vectors
        ras_b_mat = np.genfromtxt(self.inputs.aligned_vectors, delimiter="\t")
        all_bvecs = np.row_stack(
            [np.zeros(3), np.delete(ras_b_mat[:, 0:3], self.inputs.b0_indices,
                                    axis=0)]
        )
        all_bvals = np.concatenate(
            [np.zeros(1), np.delete(ras_b_mat[:, 3], self.inputs.b0_indices)]
        )

        # Which sample points are too close to the one we want to predict?
        training_mask = _nonoverlapping_qspace_samples(
            pred_val, pred_vec, all_bvals, all_bvecs,
            self.inputs.minimal_q_distance
        )
        training_indices = np.flatnonzero(training_mask[1:])
        training_image_paths = [self.inputs.b0_median] + [
            all_images[idx] for idx in training_indices
        ]
        training_bvecs = all_bvecs[training_mask]
        training_bvals = all_bvals[training_mask]
        # print(f"Training with volumes: {training_indices}")

        # Build gradient table object
        # Here, B0_THRESHOLD is imposed BY US, so we set it at 0.
        training_gtab = gradient_table_from_bvals_bvecs(
            training_bvals, training_bvecs, b0_threshold=0
        )

        # Checked shelled-ness
        if len(np.unique(training_gtab.bvals)) > 2:
            is_shelled = True
        else:
            is_shelled = False

        # Get the vector for the desired coordinate
        # Again, here, B0_THRESHOLD is imposed BY US, so we set it at 0.
        prediction_gtab = gradient_table_from_bvals_bvecs(
            np.array(pred_val)[None], np.array(pred_vec)[None, :],
            b0_threshold=0
        )

        val_str = str((pred_val,) +
                      tuple(np.round(pred_vec, decimals=2))
                      ).replace(', ', '_').replace(
            '(', '').replace(')', '')
        if is_shelled and self.inputs.model_name == "3dshore":
            # This part is adapted from Matt's old code for multi-shell Shore

            radial_order = 6
            zeta = 700
            lambdaN = 1e-8
            lambdaL = 1e-8
            t1 = time.time()
            estimator_shore = ShoreModel(training_gtab,
                                         radial_order=radial_order,
                                         zeta=zeta, lambdaN=lambdaN,
                                         lambdaL=lambdaL)
            estimator_shore_fit = estimator_shore.fit(
                series_files2series_arr(training_image_paths), mask=mask_array)
            t2 = time.time()
            print(f"Fit time: {t2 - t1}")
            t3 = time.time()
            pred_shore_fit = estimator_shore_fit.predict(prediction_gtab)
            t4 = time.time()
            print(f"Predict time: {t4 - t3}")
            pred_shore_fit[~mask_array] = 0
            pred_fit_file = f"{runtime.cwd}/predicted_shore_" \
                                  f"{val_str}.nii.gz"
            nb.Nifti1Image(pred_shore_fit, mask_img.affine, mask_img.header
                           ).to_filename(pred_fit_file)
        elif self.inputs.model_name == "sfm":
            t1 = time.time()
            sfm_all = SFM4HMC(
                training_gtab,
                isotropic=ExponentialIsotropicModel)

            sff, _ = sfm_all.fit(series_files2series_arr(training_image_paths),
                                 alpha=10e-10, mask=mask_array, tol=10e-10,
                                 iso_params=None)
            t2 = time.time()
            print(f"Fit time: {t2 - t1}")
            t3 = time.time()
            pred_sfm_fit = sff.predict(prediction_gtab,
                                       S0=np.array(nb.load(
                                           self.inputs.b0_median).dataobj))
            t4 = time.time()
            print(f"Predict time: {t4 - t3}")

            pred_fit_file = f"{runtime.cwd}/predicted_" \
                            f"{val_str}.nii.gz"
            pred_sfm_fit[~mask_array] = 0

            nb.Nifti1Image(pred_sfm_fit, mask_img.affine, mask_img.header
                           ).to_filename(pred_fit_file)
        elif self.inputs.model_name == "tensor":

            t1 = time.time()
            estimator_ten = TensorModel(training_gtab)
            estimator_ten_fit = estimator_ten.fit(
                series_files2series_arr(training_image_paths), mask=mask_array)
            t2 = time.time()
            print(f"Fit time: {t2 - t1}")
            t3 = time.time()
            pred_ten_fit = estimator_ten_fit.predict(prediction_gtab)[..., 0]
            t4 = time.time()
            print(f"Predict time: {t4 - t3}")
            pred_ten_fit[~mask_array] = 0
            pred_fit_file = f"{runtime.cwd}/predicted_" \
                            f"{val_str}.nii.gz"
            nb.Nifti1Image(pred_ten_fit, mask_img.affine, mask_img.header
                           ).to_filename(pred_fit_file)
        # elif self.inputs.model_name == "ensemble":
        #     from mlens.ensemble import SuperLearner
        #
        #     # Use tensor as an initiator of a meta-learner
        #     ensemble = SuperLearner()
        #
        #     t1 = time.time()
        #     # Instantiate tensor
        #     estimator_ten = TensorModel(training_gtab, mask=mask_array)
        #
        #     # Instantiate sfm
        #     sfm_all = SFM4HMC(
        #         training_gtab,
        #         isotropic=ExponentialIsotropicModel)
        #
        #     sff, _ = sfm_all.fit(training_data, alpha=10e-10,
        #                                       mask=mask_array,
        #                                       tol=10e-10, iso_params=None)
        #
        #     t2 = time.time()
        #     print(f"Fit time: {t2 - t1}")
        #     t3 = time.time()
        #
        #     ensemble.add([estimator_ten, sff])
        #
        #     pred_fit = ensemble.predict(prediction_gtab)[..., 0]
        #     t4 = time.time()
        #     print(f"Predict time: {t4 - t3}")
        #
        #     pred_fit[~mask_array] = 0
        #     pred_fit_file = f"{runtime.cwd}/predicted_" \
        #                     f"{(pred_val,) + tuple(np.round(pred_vec, decimals=2))}.nii.gz"
        #     nb.Nifti1Image(pred_fit, mask_img.affine, mask_img.header
        #                    ).to_filename(pred_fit_file)
        else:
            raise ValueError("Model not supported.")

        self._results["predicted_image"] = pred_fit_file

        return runtime
