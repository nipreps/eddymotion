# Data for unit and integration tests

This folder contains several useful data sets for testing.

## Realistic head motion parameters: `head-motion-parameters.aff12.1D`

This file contains head motion parameters estimated on a functional MRI dataset, using OpenNeuro's [ds005](https://openneuro.org/datasets/ds000005/versions/00001).
In particular, `ds000005/sub-01/func/sub-01_task-mixedgamblestask_run-01_bold.nii.gz` is used.

The command line executed is:
```
3dvolreg -prefix /tmp/sub-01_task-mixedgamblestask_run-01_desc-motioncorr_bold.nii.gz -base 120 -twopass -1Dmatrix_save eddymotion/tests/data/head-motion-parameters /data/ds000005/sub-01/func/sub-01_task-mixedgamblestask_run-01_bold.nii.gz
```
