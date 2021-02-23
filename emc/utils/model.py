import numpy as np
from dipy.reconst.sfm import (SparseFascicleModel, SparseFascicleFit,
                              IsotropicFit, IsotropicModel,
                              _to_fit_iso, nanmean)
import warnings
from functools import partial
from scipy.linalg import svd

svd = partial(svd, full_matrices=False)

BIG_BIAS = 10e3
SMALL_BIAS = 10e-3
BIAS_STEP = 0.2


def _do_svd(X, y, jit=True):
    """
    Helper function to produce SVD outputs
    """
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    if X.shape[0] > X.shape[1]:
        uu, ss, v_t = svd(X.T @ X)
        selt = np.sqrt(ss)
        if y.shape[-1] >= X.shape[0]:
            ynew = (1/selt) @ v_t @ X.T @ y
        else:
            ynew = np.diag(1./selt) @ v_t @ (X.T @ y)

    else:
        uu, selt, v_t = svd(X)
        # This rotates the targets by the unitary matrix uu.T:
        ynew = uu.T @ y

    ols_coef = (ynew.T / selt).T

    return uu, selt, v_t, ols_coef


class ExponentialIsotropicModel(IsotropicModel):
    """
    Representing the isotropic signal as a fit to an exponential decay function
    with b-values
    """
    def fit(self, data, mask=None):
        """

        Parameters
        ----------
        data : ndarray

        Returns
        -------
        ExponentialIsotropicFit class instance.
        """
        to_fit = _to_fit_iso(data, self.gtab, mask=mask)
        # Fitting to the log-transformed relative data is much faster:
        nz_idx = to_fit > 0
        to_fit[nz_idx] = np.log(to_fit[nz_idx])
        to_fit[~nz_idx] = -np.inf
        p = nanmean(to_fit / self.gtab.bvals[~self.gtab.b0s_mask], -1)
        params = -p
        if mask is None:
            params = np.reshape(params, data.shape[:-1])
        else:
            out_params = np.zeros(data.shape[:-1])
            out_params[mask] = params
            params = out_params
        return ExponentialIsotropicFit(self, params)


class ExponentialIsotropicFit(IsotropicFit):
    """
    A fit to the ExponentialIsotropicModel object, based on data.
    """
    def predict(self, gtab=None):
        """
        Predict the isotropic signal, based on a gradient table. In this case,
        the prediction will be for an exponential decay with the mean
        diffusivity derived from the data that was fit.

        Parameters
        ----------
        gtab : a GradientTable class instance (optional)
            Defaults to use the gtab from the IsotropicModel from which this
            fit was derived.
        """
        if gtab is None:
            gtab = self.model.gtab
        if len(self.params.shape) == 0:
            pred = np.exp(-gtab.bvals[~gtab.b0s_mask] *
                          (np.zeros(np.sum(~gtab.b0s_mask)) +
                          self.params[..., np.newaxis]))
        else:
            pred = np.exp(-gtab.bvals[~gtab.b0s_mask] *
                          (np.zeros((self.params.shape +
                                     (int(np.sum(~gtab.b0s_mask)), ))) +
                          self.params[..., np.newaxis]))
        return pred


class SFM4HMC(SparseFascicleModel):
    """
    We need to reimplement the fit, so that we can use the FRR cleverness
    under the hood
    """
    def fit(self, data, alpha=0.1, mask=None, tol=10e-10, iso_params=None):
        """
        Fit the SparseFascicleModel object to data.

        Parameters
        ----------
        data : array
            The measured signal.

        mask : array, optional
            A boolean array used to mark the coordinates in the data that
            should be analyzed. Has the shape `data.shape[:-1]`. Default: None,
            which implies that all points should be analyzed.

        Returns
        -------
        SparseFascicleFit object
        """

        data_in_mask = data[mask]
        # Fitting is done on the relative signal (S/S0):
        flat_S0 = np.mean(data_in_mask[..., self.gtab.b0s_mask], -1)
        if not flat_S0.size or not flat_S0.max():
            flat_S = np.zeros(data_in_mask[..., ~self.gtab.b0s_mask].shape)
        else:
            flat_S = (data_in_mask[..., ~self.gtab.b0s_mask] /
                      flat_S0[..., None])

        if iso_params is None:
            isotropic = self.isotropic(self.gtab).fit(data, mask)
        else:
            isotropic = ExponentialIsotropicFit(self.isotropic(self.gtab),
                                                iso_params)

        isopredict = isotropic.predict()

        if mask is None:
            isopredict = np.reshape(isopredict, (-1, isopredict.shape[-1]))
        else:
            isopredict = isopredict[mask]

        # Here's where things get different: ##
        y = (flat_S - isopredict).T
        # Making sure nan voxels get 0 params:
        nan_targets = np.unique(np.where(~np.isfinite(y))[1])
        y[:, nan_targets] = 0

        ### FIT FRACRIDGE
        uu, selt, v_t, ols_coef = _do_svd(self.design_matrix, y)
        # Set solutions for small eigenvalues to 0 for all targets:
        isbad = selt < tol
        if np.any(isbad):
            warnings.warn("Some eigenvalues are being treated as 0")

        ols_coef[isbad, ...] = 0
        seltsq = selt**2
        sclg = seltsq / (seltsq + alpha)
        coef = sclg[:, np.newaxis] * ols_coef
        coef = v_t.T @ coef

        flat_params = coef.squeeze().T

        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            beta = flat_params.reshape(out_shape)
            S0 = flat_S0.reshape(data.shape[:-1])
        else:
            beta = np.zeros(data.shape[:-1] +
                            (self.design_matrix.shape[-1],))
            beta[mask, :] = flat_params
            S0 = np.zeros(data.shape[:-1])
            S0[mask] = flat_S0

        return SparseFascicleFit(self, beta, S0, isotropic), isotropic.params
