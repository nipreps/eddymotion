*Eddymotion*
============
**Estimating head-motion and deformations derived from eddy-currents in diffusion MRI data**.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4680599.svg
   :target: https://doi.org/10.5281/zenodo.4680599
   :alt: DOI

Retrospective estimation of head-motion between diffusion-weighted images (DWI) acquired within
diffusion MRI (dMRI) experiments renders exceptionally challenging1 for datasets including
high-diffusivity (or “high b”) images.
These “high b” (b > 1000s/mm2) DWIs enable higher angular resolution, as compared to more traditional
diffusion tensor imaging (DTI) schemes.
UNDISTORT [1]_ (Using NonDistorted Images to Simulate a Template Of the Registration Target)
was the earliest method addressing this issue, by simulating a target DW image without motion
or distortion from a DTI (b=1000s/mm2) scan of the same subject.
Later, Andersson and Sotiropoulos [2]_ proposed a similar approach (widely available within the
FSL ``eddy`` tool), by predicting the target DW image to be registered from the remainder of the
dMRI dataset and modeled with a Gaussian process.
Besides the need for less data, ``eddy`` has the advantage of implicitly modeling distortions due
to Eddy currents.
More recently, Cieslak et al. [3]_ integrated both approaches in *SHORELine*, by
(i) setting up a leave-one-out prediction framework as in eddy; and
(ii) replacing eddy’s general-purpose Gaussian process prediction with the SHORE [4]_ diffusion model.

*Eddymotion* is an open implementation of eddy-current and head-motion correction that builds upon
the work of ``eddy`` and *SHORELine*, while generalizing these methods to multiple acquisition schemes
(single-shell, multi-shell, and diffusion spectrum imaging) using diffusion models available with DIPY [5]_.


.. image:: docs/_static/eddymotion-flowchart.svg
   :alt: The eddymotion flowchart


.. [1] S. Ben-Amitay et al., Motion correction and registration of high b-value diffusion weighted images, Magnetic
   Resonance in Medicine 67:1694–1702 (2012)
.. [2] J. L. R. Andersson. et al., An integrated approach to correction for off-resonance effects and subject movement
   in diffusion MR imaging, NeuroImage 125 (2016) 1063–1078
.. [3] M. Cieslak et al., QSIPrep: An integrative platform for preprocessing and reconstructing diffusion MRI data.
   Nature Methods, 18(7), 775–778 (2021)
.. [4] E. Ozarslan et al., Simple Harmonic Oscillator Based Reconstruction and Estimation for Three-Dimensional Q-Space
   MRI. in Proc. Intl. Soc. Mag. Reson. Med. vol. 17 1396 (2009)
.. [5] E. Garyfallidis et al., Dipy, a library for the analysis of diffusion MRI data. Front. Neuroinformatics 8, 8
   (2014)
