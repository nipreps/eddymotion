.. include:: links.rst

How to Run
============

Using Eddymotion in a Python Module
-------------------------------------

To incorporate Eddymotion functionalities into a Python module, follow these steps:

1. **Import Eddymotion Components**: Begin by importing the necessary components from the Eddymotion package into your Python module:

   .. code-block:: python

      from eddymotion.data.dmri import DWI
      from eddymotion.estimator import EddyMotionEstimator

2. **Load DWI Data**: Load diffusion MRI (dMRI) data into a `DWI` object using the `load` function. If your data are provided in a NIfTI file format, ensure to provide a file containing the gradient information with the argument "gradients_file":

   .. code-block:: python

      dwi_data = DWI.load('/path/to/your/dwi_data.nii.gz', gradients_file='/path/to/your/gradient_file')

3. **Instantiate an Eddymotion Estimator Object**: Create an instance of the `EddyMotionEstimator` class. This object encapsulates the tools required to estimate rigid-body head-motion and distortions derived from eddy-currents.

   .. code-block:: python

      estimator = EddyMotionEstimator()

4. **Fit the Model to Estimate the Affine Transformation**:

   Utilize the `fit` method of the `EddyMotionEstimator` object to estimate the affine transformation parameters:

   .. code-block:: python

      estimated_affine = estimator.fit(
          dwi_dataset,
          align_kwargs=align_kwargs,
          models=models,
          omp_nthreads=omp_nthreads,
          n_jobs=n_jobs,
          seed=seed,
      )

   In the `fit` method, the Leave-One-Volume-Out (LOVO) splitting technique is employed to iteratively process DWI data volumes for each specified model. Affine transformations are estimated to align the volumes, updating the `DWI` object with the estimated parameters. This method accepts several parameters:

   - `dwi_dataset`: The target DWI dataset, represented by this tool's internal type.
   - `align_kwargs`: Parameters to configure the image registration process.
   - `models`: Selects the diffusion model that will be used to generate the registration target corresponding to each gradient map.
   - `omp_nthreads`: Maximum number of threads an individual process may use.
   - `n_jobs`: Number of parallel jobs.
   - `seed`: Seed for the random number generator (necessary for deterministic estimation).

   The method returns a list of affine matrices encoding the estimated parameters of the deformations caused by head-motion and eddy-currents.

   Example:

   .. code-block:: python

      estimated_affine = estimator.fit(
          dwi_dataset,
          models=["b0"],
          omp_nthreads=4,
          n_jobs=4,
          seed=42,
      )

5. **Save Results**: After estimating the transformations, save the realigned DWI data to your preferred output format, either HDF5 or NIfTI:

   .. code-block:: python

      output_filename = '/path/to/save/your/output.h5'
      dwi_data.to_filename(output_filename)

   or as a NIfTI file:

   .. code-block:: python

     output_filename = '/path/to/save/your/output.nii.gz'
     dwi_data.to_nifti(output_filename)

6. **Plotting**: Visualize your data and results using built-in plotting functions of the DWI objects:
   - Use `plot_mosaic` to visualize one direction of the dMRI dataset.
   - Employ `plot_gradients` to visualize diffusion gradients.

Example Usage:

```python
# Visualize DWI data at a specific index
dwi_data.plot_mosaic(index=0)

# Visualize gradients
dwi_data.plot_gradients()

Command-Line Arguments
----------------------
*eddymotion* 
