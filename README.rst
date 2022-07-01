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
UNDISTORT1 (Using NonDistorted Images to Simulate a Template Of the Registration Target)
was the earliest method addressing this issue, by simulating a target DW image without motion
or distortion from a DTI (b=1000s/mm2) scan of the same subject.
Later, Andersson and Sotiropoulos2 proposed a similar approach (widely available within the 
FSL ``eddy`` tool), by predicting the target DW image to be registered from the remainder of the
dMRI dataset and modeled with a Gaussian process.
Besides the need for less data, ``eddy`` has the advantage of implicitly modeling distortions due
to Eddy currents.
More recently, Cieslak et al.3 integrated both approaches in *SHORELine*, by
(i) setting up a leave-one-out prediction framework as in eddy; and
(ii) replacing eddy’s general-purpose Gaussian process prediction with the SHORE4 diffusion model.

*Eddymotion* is an open implementation of eddy-current and head-motion correction that builds upon
the work of ``eddy`` and *SHORELine*, while generalizing these methods to multiple acquisition schemes
(single-shell, multi-shell, and diffusion spectrum imaging) using diffusion models available with DIPY5.


.. image:: docs/_static/eddymotion-flowchart.svg
   :alt: The eddymotion flowchart