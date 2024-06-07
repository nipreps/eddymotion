.. include:: links.rst

How to Use
==========
Incorporating Eddymotion into a Python module or script
-------------------------------------------------------
To utilize Eddymotion functionalities within your Python module or script, follow these steps:

1. **Import Eddymotion Components**: Start by importing necessary components from the Eddymotion package:

   .. code-block:: python

      # Import required components from the Eddymotion package
      from eddymotion.data import dmri
      from eddymotion.estimator import EddyMotionEstimator

2. **Load DWI Data**: Load diffusion MRI (dMRI) data into a `DWI` object using the `load` function. Ensure the gradient table is provided. It should have one row per diffusion-weighted image. The first three columns represent the gradient directions, and the last column indicates the timing and strength of the gradients in units of s/mm² [ R A S+ b ]. If your data are in NIfTI file format, include a file containing the gradient information with the argument "gradients_file":

   .. code-block:: python

      # Load dMRI data into a DWI object
      dwi_data = dmri.load('/path/to/your/dwi_data.nii.gz', gradients_file='/path/to/your/gradient_file')

   .. note::

      To run the examples and tests from this page,
      find `sample data <https://osf.io/download/6at98/>`__ on OSF.
      To load from an HDF5 file, use:

      .. code-block:: python

         dwi_data = dmri.DWI.from_filename('/path/to/downloaded/dwi_full.h5')


3. **Instantiate an Eddymotion Estimator Object**: Create an instance of the `EddyMotionEstimator` class, which encapsulates tools for estimating rigid-body head motion and distortions due to eddy currents.

   .. code-block:: python

      # Create an instance of the EddyMotionEstimator class
      estimator = EddyMotionEstimator()

4. **Fit the Models to Estimate the Affine Transformation**:

   Use the `estimate` method of the `EddyMotionEstimator` object to estimate the affine transformation parameters:

   .. code-block:: python

      # Estimate affine transformation parameters
      estimated_affine = estimator.estimate(
          dwi_data,
          align_kwargs=align_kwargs,
          models=models,
          omp_nthreads=omp_nthreads,
          n_jobs=n_jobs,
          seed=seed,
      )

   The `estimate` method employs the Leave-One-Volume-Out (LOVO) splitting technique to iteratively process DWI data volumes for each specified model. Affine transformations align the volumes, updating the `DWI` object with the estimated parameters. This method accepts several parameters:

   - `dwi_data`: The target DWI dataset, represented by this tool's internal type.
   - `align_kwargs`: Parameters to configure the image registration process.
   - `models`: list of diffusion models used to generate the registration target for each gradient map. For a list of available models, see :doc:`api/eddymotion.model`.
   - `omp_nthreads`: Maximum number of threads an individual process may use.
   - `n_jobs`: Number of parallel jobs.
   - `seed`: Seed for the random number generator (necessary for deterministic estimation).

   The method returns an Nx4x4 array of affine matrices encoding the estimated parameters of
   the deformations due to head motion and eddy currents.

   Example:

   .. code-block:: python

      # Example of fitting the model
      estimated_affine = estimator.estimate(
          dwi_data,
          models=["b0"],
          omp_nthreads=4,
          n_jobs=4,
          seed=42,
      )

5. **Save Results**: Once transformations are estimated, save the realigned DWI data in your preferred output format, either HDF5 or NIfTI:

   .. code-block:: python

      # Save realigned DWI data in HDF5 format
      output_filename = '/path/to/save/your/output.h5'
      dwi_data.to_filename(output_filename)

   or as a NIfTI file:

   .. code-block:: python

     # Save realigned DWI data in NIfTI format
     output_filename = '/path/to/save/your/output.nii.gz'
     dwi_data.to_nifti(output_filename)

6. **Plotting**: Visualize data and results using built-in plotting functions of the DWI objects:

   - Use `plot_mosaic` to visualize one direction of the dMRI dataset.
   - Employ `plot_gradients` to visualize diffusion gradients.

   Example Usage:

   .. code-block:: python

      # Visualize DWI data at a specific index
      dwi_data.plot_mosaic(index=0)
      # Visualize gradients
      dwi_data.plot_gradients()
