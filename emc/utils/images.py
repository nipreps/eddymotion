"""
Image prediction tools
"""
import numpy as np
import nibabel as nb
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from nipype.utils.filemanip import fname_presuffix


def _list_squeeze(in_list):
    return [item[0] for item in in_list]


def flatten(l):
    """
    Flatten list of lists.
    """
    import collections

    for el in l:
        if isinstance(
                el, collections.Iterable) and not isinstance(
                el, (str, bytes)):
            for ell in flatten(el):
                yield ell
        else:
            yield el


def _pass_predicted_outs(ins):
    import os
    from emc.utils.images import flatten

    return [i for i in list(flatten(ins)) if i is not '/' and
            os.path.isfile(i)]


def mask_4d(dwi_file, mask_file):
    from nipype.utils.filemanip import fname_presuffix

    dwi_masked = fname_presuffix(
        dwi_file,
        use_ext=False,
        suffix="_masked.nii.gz",
    )

    dwi_img = nb.load(dwi_file)
    dwi_data = dwi_img.get_fdata()
    mask_data = nb.load(mask_file).get_fdata()
    data_in_mask = np.nan_to_num(np.broadcast_to(mask_data[..., None],
                                                 dwi_data.shape
                                                 ).astype('bool') * dwi_data)

    nb.Nifti1Image(data_in_mask, affine=dwi_img.affine).to_filename(dwi_masked)
    return dwi_masked


def average_images(images):
    from nilearn.image import mean_img

    average_img = mean_img([nb.load(img) for img in images])
    output_average_image = fname_presuffix(
        images[0], use_ext=False, suffix="_mean.nii.gz"
    )
    average_img.to_filename(output_average_image)
    return output_average_image


def series_files2series_arr(image_list, dtype=np.float32):
    output_array = np.zeros(
        tuple(nb.load(image_list[0]).shape) + (len(image_list),)).astype(dtype=dtype)
    for image_num, image_path in enumerate(image_list):
        output_array[..., image_num] = np.asarray(nb.load(image_path
                                                          ).dataobj).astype(dtype=dtype)
    return output_array


def match_transforms(dwi_files, transforms, b0_indices):
    original_b0_indices = np.array(b0_indices)
    num_dwis = len(dwi_files)
    num_transforms = len(transforms)

    if num_dwis == num_transforms:
        return transforms

    # Do sanity checks
    if not len(transforms) == len(b0_indices):
        raise Exception("Number of transforms does not match number of b0 "
                        "images")

    # Create a list of which emc affines that correspond to the split images
    nearest_affines = []
    for index in range(num_dwis):
        nearest_b0_num = np.argmin(np.abs(index - original_b0_indices))
        this_transform = transforms[nearest_b0_num]
        nearest_affines.append(this_transform)

    return nearest_affines


def save_4d_to_3d(in_file):
    in_img = nb.load(in_file)
    if len(in_img.shape) > 3 or (len(in_img.shape) == 4 and
                                 in_img.shape[-1] == 1):
        files_3d = nb.four_to_three(in_img)
        out_files = []
        for i, file_3d in enumerate(files_3d):
            out_file = fname_presuffix(in_file, suffix="_tmp_{}".format(i))
            file_3d.to_filename(out_file)
            out_files.append(out_file)
        del files_3d
    else:
        out_file = fname_presuffix(in_file, suffix="_tmp_{}".format(0))
        in_img.to_filename(out_file)
        out_files = [out_file]
    in_img.uncache()

    return out_files


def prune_b0s_from_dwis(in_files, b0_ixs):
    """
    Remove *b0* volume files from a complete list of DWI volume files.

    Parameters
    ----------
    in_files : list
        A list of NIfTI file paths corresponding to each 3D volume of a
        DWI image (i.e. including B0's).
    b0_ixs : list
        List of B0 indices.

    Returns
    -------
    out_files : list
       A list of file paths to 3d NIFTI images.

    Examples
    --------
    >>> os.chdir(tmpdir)
    >>> b0_ixs = np.where(np.loadtxt(str(dipy_datadir / "HARDI193.bval")) <= 50)[0].tolist()[:2]
    >>> in_file = str(dipy_datadir / "HARDI193.nii.gz")
    >>> threeD_files = save_4d_to_3d(in_file)
    >>> out_files = prune_b0s_from_dwis(threeD_files, b0_ixs)
    >>> assert sum([os.path.isfile(i) for i in out_files]) == len(out_files)
    >>> assert len(out_files) == len(threeD_files) - len(b0_ixs)
    """
    if in_files[0].endswith("_warped.nii.gz"):
        out_files = [
            i
            for j, i in enumerate(
                sorted(
                    in_files, key=lambda x: int(x.split("_")[-2].split(".nii.gz")[0])
                )
            )
            if j not in b0_ixs
        ]
    else:
        out_files = [
            i
            for j, i in enumerate(
                sorted(
                    in_files, key=lambda x: int(x.split("_")[-1].split(".nii.gz")[0])
                )
            )
            if j not in b0_ixs
        ]
    return out_files


def save_3d_to_4d(in_files):
    img_4d = nb.funcs.concat_images([nb.load(img_3d) for img_3d in in_files])
    out_file = fname_presuffix(in_files[0], suffix="_merged")
    img_4d.to_filename(out_file)
    del img_4d
    return out_file


def get_params(A):
    """This is a copy of spm's spm_imatrix where
    we already know the rotations and translations matrix,
    shears and zooms (as outputs from fsl FLIRT/avscale)
    Let A = the 4x4 rotation and translation matrix
    R = [          c5*c6,           c5*s6, s5]
        [-s4*s5*c6-c4*s6, -s4*s5*s6+c4*c6, s4*c5]
        [-c4*s5*c6+s4*s6, -c4*s5*s6-s4*c6, c4*c5]
    """

    def rang(b):
        a = min(max(b, -1), 1)
        return a

    Ry = np.arcsin(A[0, 2])
    # Rx = np.arcsin(A[1, 2] / np.cos(Ry))
    # Rz = np.arccos(A[0, 1] / np.sin(Ry))

    if (abs(Ry) - np.pi / 2) ** 2 < 1e-9:
        Rx = 0
        Rz = np.arctan2(-rang(A[1, 0]), rang(-A[2, 0] / A[0, 2]))
    else:
        c = np.cos(Ry)
        Rx = np.arctan2(rang(A[1, 2] / c), rang(A[2, 2] / c))
        Rz = np.arctan2(rang(A[0, 1] / c), rang(A[0, 0] / c))

    rotations = [Rx, Ry, Rz]
    translations = [A[0, 3], A[1, 3], A[2, 3]]

    return rotations, translations


def _vol_split(train, vol_idx):
    """ Split the 3D volumes into the train and test set.
    Parameters
    ----------
    train : ndarray
        Array of all 3D patches flattened out to be 2D.
    vol_idx: int
        The volume number that needs to be held out for training.
    Returns
    --------
    cur_x : 2D-array (nvolumes*patch_size) x (nvoxels)
        Array of patches corresponding to all the volumes except for the
        held-out volume.
    y : 1D-array
        Array of patches corresponding to the volume that is used a target for
        denoising.
    """
    # Hold-out the target volume
    mask = np.zeros(train.shape[0])
    mask[vol_idx] = 1
    cur_x = train[mask == 0]
    cur_x = cur_x.reshape(((train.shape[0]-1)*train.shape[1],
                           train.shape[2]))

    # Center voxel of the selected block
    y = train[vol_idx, train.shape[1]//2, :]
    return cur_x, y


def _vol_denoise(train, vol_idx, model, data_shape, alpha):
    """ Denoise a single 3D volume using a train and test phase.
    Parameters
    ----------
    train : ndarray
        Array of all 3D patches flattened out to be 2D.
    vol_idx : int
        The volume number that needs to be held out for training.
    model : string, or initialized linear model object.
            This will determine the algorithm used to solve the set of linear
            equations underlying this model. If it is a string it needs to be
            one of the following: {'ols', 'ridge', 'lasso'}. Otherwise,
            it can be an object that inherits from
            `dipy.optimize.SKLearnLinearSolver` or an object with a similar
            interface from Scikit-Learn:
            `sklearn.linear_model.LinearRegression`,
            `sklearn.linear_model.Lasso` or `sklearn.linear_model.Ridge`
            and other objects that inherit from `sklearn.base.RegressorMixin`.
            Default: 'ridge'.
    data_shape : ndarray
        The 4D shape of noisy DWI data to be denoised.
    alpha : float, optional
        Regularization parameter only for ridge and lasso regression models.
        default: 1.0
    Returns
    --------
    model prediction : ndarray
        Denoised array of all 3D patches flattened out to be 2D corresponding
        to the held out volume `vol_idx`.
    """
    # To add a new model, use the following API
    # We adhere to the following options as they are used for comparisons
    if model.lower() == 'ols':
        model = linear_model.LinearRegression(copy_X=False)

    elif model.lower() == 'ridge':
        model = linear_model.Ridge(copy_X=False, alpha=alpha)

    elif model.lower() == 'lasso':
        model = linear_model.Lasso(copy_X=False, max_iter=50, alpha=alpha)
    else:
        e_s = "The `solver` key-word argument needs to be: "
        e_s += "'ols', 'ridge', 'lasso' or a "
        e_s += "`dipy.optimize.SKLearnLinearSolver` object"
        raise ValueError(e_s)

    cur_x, y = _vol_split(train, vol_idx)
    model.fit(cur_x.T, y.T)

    return model.predict(cur_x.T).reshape(data_shape[0], data_shape[1],
                                          data_shape[2])


def _extract_3d_patches(arr, patch_radius):
    """ Extract 3D patches from 4D DWI data.
    Parameters
    ----------
    arr : ndarray
        The 4D noisy DWI data to be denoised.
    patch_radius : int or 1D array
        The radius of the local patch to be taken around each voxel (in
        voxels).
    Returns
    --------
    all_patches : ndarray
        All 3D patches flattened out to be 2D corresponding to the each 3D
        volume of the 4D DWI data.
    """
    if isinstance(patch_radius, int):
        patch_radius = np.ones(3, dtype=int) * patch_radius
    if len(patch_radius) != 3:
        raise ValueError("patch_radius should have length 3")
    else:
        patch_radius = np.asarray(patch_radius, dtype=int)
    patch_size = 2 * patch_radius + 1

    dim = arr.shape[-1]

    all_patches = []

    # loop around and find the 3D patch for each direction
    for i in range(patch_radius[0], arr.shape[0] -
                   patch_radius[0], 1):
        for j in range(patch_radius[1], arr.shape[1] -
                       patch_radius[1], 1):
            for k in range(patch_radius[2], arr.shape[2] -
                           patch_radius[2], 1):

                ix1 = i - patch_radius[0]
                ix2 = i + patch_radius[0] + 1
                jx1 = j - patch_radius[1]
                jx2 = j + patch_radius[1] + 1
                kx1 = k - patch_radius[2]
                kx2 = k + patch_radius[2] + 1

                X = arr[ix1:ix2, jx1:jx2,
                        kx1:kx2].reshape(np.prod(patch_size), dim)
                all_patches.append(X)

    return np.array(all_patches).T


def patch2self(data, bvals, patch_radius=[0, 0, 0], model='ridge',
               b0_threshold=50, out_dtype=None, alpha=1.0, verbose=False,
               b0_denoising=True, clip_negative_vals=True,
               shift_intensity=False):
    """ Patch2Self Denoiser
    Parameters
    ----------
    data : ndarray
        The 4D noisy DWI data to be denoised.
    bvals : 1D array
        Array of the bvals from the DWI acquisition
    patch_radius : int or 1D array, optional
        The radius of the local patch to be taken around each voxel (in
        voxels). Default: 0 (denoise in blocks of 1x1x1 voxels).
    model : string, or initialized linear model object.
            This will determine the algorithm used to solve the set of linear
            equations underlying this model. If it is a string it needs to be
            one of the following: {'ols', 'ridge', 'lasso'}. Otherwise,
            it can be an object that inherits from
            `dipy.optimize.SKLearnLinearSolver` or an object with a similar
            interface from Scikit-Learn:
            `sklearn.linear_model.LinearRegression`,
            `sklearn.linear_model.Lasso` or `sklearn.linear_model.Ridge`
            and other objects that inherit from `sklearn.base.RegressorMixin`.
            Default: 'ridge'.
    b0_threshold : int, optional
        Threshold for considering volumes as b0.
    out_dtype : str or dtype, optional
        The dtype for the output array. Default: output has the same dtype as
        the input.
    alpha : float, optional
        Regularization parameter only for ridge regression model.
        Default: 1.0
    verbose : bool, optional
        Show progress of Patch2Self and time taken.
    b0_denoising : bool, optional
        Skips denoising b0 volumes if set to False.
        Default: True
    clip_negative_vals : bool, optional
        Sets negative values after denoising to 0 using `np.clip`.
        Default: True
    shift_intensity : bool, optional
        Shifts the distribution of intensities per volume to give
        non-negative values
        Default: False
    Returns
    --------
    denoised array : ndarray
        This is the denoised array of the same size as that of the input data,
        clipped to non-negative values.
    noise_level_image : ndarray
        RMS of the estimated noise across all images. A 3D matrix.
    References
    ----------
    [patch2self] S. Fadnavis, J. Batson, E. Garyfallidis, Patch2Self:
                 Denoising Diffusion MRI with Self-supervised Learning,
                 Advances in Neural Information Processing Systems 33 (2020)
    """
    import time

    patch_radius = np.asarray(patch_radius, dtype=np.int)

    if not data.ndim == 4:
        raise ValueError("Patch2Self can only denoise on 4D arrays.",
                         data.shape)

    if data.shape[3] < 10:
        print("The intput data has less than 10 3D volumes. Patch2Self may not",
             "give denoising performance.")

    if out_dtype is None:
        out_dtype = data.dtype

    # We retain float64 precision, iff the input is in this precision:
    if data.dtype == np.float64:
        calc_dtype = np.float64

    # Otherwise, we'll calculate things in float32 (saving memory)
    else:
        calc_dtype = np.float32

    # Segregates volumes by b0 threshold
    b0_idx = np.argwhere(bvals <= b0_threshold)
    dwi_idx = np.argwhere(bvals > b0_threshold)

    data_b0s = np.squeeze(np.take(data, b0_idx, axis=3))
    data_dwi = np.squeeze(np.take(data, dwi_idx, axis=3))

    # create empty arrays
    denoised_b0s = np.empty((data_b0s.shape), dtype=calc_dtype)
    denoised_dwi = np.empty((data_dwi.shape), dtype=calc_dtype)

    denoised_arr = np.empty((data.shape), dtype=calc_dtype)

    if verbose:
        t1 = time.time()

    # if only 1 b0 volume, skip denoising it
    if data_b0s.ndim == 3 or not b0_denoising:
        if verbose:
            print("b0 denoising skipped...")
        denoised_b0s = data_b0s

    else:
        train_b0 = _extract_3d_patches(np.pad(data_b0s, ((patch_radius[0],
                                              patch_radius[0]),
                                              (patch_radius[1],
                                               patch_radius[1]),
                                              (patch_radius[2],
                                               patch_radius[2]),
                                              (0, 0)), mode='constant'),
                                       patch_radius=patch_radius)

        for vol_idx in range(0, data_b0s.shape[3]):
            denoised_b0s[..., vol_idx] = _vol_denoise(train_b0,
                                                      vol_idx, model,
                                                      data_b0s.shape,
                                                      alpha=alpha)

            if verbose:
                print("Denoised b0 Volume: ", vol_idx)

    # Separate denoising for DWI volumes
    train_dwi = _extract_3d_patches(np.pad(data_dwi, ((patch_radius[0],
                                                       patch_radius[0]),
                                                      (patch_radius[1],
                                                       patch_radius[1]),
                                                      (patch_radius[2],
                                                       patch_radius[2]),
                                                      (0, 0)),
                                           mode='constant'),
                                    patch_radius=patch_radius)

    # Insert the separately denoised arrays into the respective empty arrays
    for vol_idx in range(0, data_dwi.shape[3]):
        denoised_dwi[..., vol_idx] = _vol_denoise(train_dwi,
                                                  vol_idx, model,
                                                  data_dwi.shape,
                                                  alpha=alpha)

        if verbose:
            print("Denoised DWI Volume: ", vol_idx)

    if verbose:
        t2 = time.time()
        print('Total time taken for Patch2Self: ', t2-t1, " seconds")

    if data_b0s.ndim == 3:
        denoised_arr[:, :, :, b0_idx[0][0]] = denoised_b0s
    else:
        for i, idx in enumerate(b0_idx):
            denoised_arr[:, :, :, idx[0]] = np.squeeze(denoised_b0s[..., i])

    for i, idx in enumerate(dwi_idx):
        denoised_arr[:, :, :, idx[0]] = np.squeeze(denoised_dwi[..., i])

    # shift intensities per volume to handle for negative intensities
    if shift_intensity and not clip_negative_vals:
        for i in range(0, denoised_arr.shape[3]):
            shift = np.min(data[..., i]) - np.min(denoised_arr[..., i])
            denoised_arr[..., i] = denoised_arr[..., i] + shift

    # clip out the negative values from the denoised output
    elif clip_negative_vals and not shift_intensity:
        denoised_arr.clip(min=0, out=denoised_arr)

    elif clip_negative_vals and shift_intensity:
        denoised_arr.clip(min=0, out=denoised_arr)

    # Calculate a "noise level" image
    noise_level_image = np.sqrt(np.mean((data - denoised_arr) ** 2, axis=3))

    return np.array(denoised_arr, dtype=out_dtype), noise_level_image