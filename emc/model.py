"""A factory class that adapts DIPY's dMRI models."""
import warnings
import numpy as np
from dipy.core.gradients import gradient_table


class ModelFactory:
    """A factory for instantiating diffusion models."""

    @staticmethod
    def init(gtab, model="TensorModel", **kwargs):
        """
        Instatiate a diffusion model.

        Parameters
        ----------
        gtab : :obj:`numpy.ndarray`
            An array representing the gradient table in RAS+B format.
        model : :obj:`str`
            Diffusion model.
            Options: ``"3DShore"``, ``"SFM"``, ``"Tensor"``, ``"S0"``

        Return
        ------
        model : :obj:`~dipy.reconst.ReconstModel`
            An model object compliant with DIPY's interface.

        """
        if model.lower() in ("s0", "b0"):
            return TrivialB0Model(gtab=gtab, S0=kwargs.pop("S0"))

        # Generate a GradientTable object for DIPY
        gtab = _rasb2dipy(gtab)
        param = {}

        if model.lower().startswith("3dshore"):
            from dipy.reconst.shore import ShoreModel as Model

            param = {
                "radial_order": 6,
                "zeta": 700,
                "lambdaN": 1e-8,
                "lambdaL": 1e-8,
            }

        elif model.lower().startswith("sfm"):
            from emc.utils.model import (
                SFM4HMC as Model,
                ExponentialIsotropicModel,
            )

            param = {
                "isotropic": ExponentialIsotropicModel,
            }

        elif model.lower().startswith("tensor"):
            Model = TensorModel

        elif model.lower().startswith("dki"):
            Model = DKIModel

        else:
            raise NotImplementedError(f"Unsupported model <{model}>.")

        param.update(kwargs)
        return Model(gtab, **param)


class TrivialB0Model:
    """
    A trivial model that returns a *b=0* map always.

    Implements the interface of :obj:`dipy.reconst.base.ReconstModel`.
    Instead of inheriting from the abstract base, this implementation
    follows type adaptation principles, as it is easier to maintain
    and to read (see https://www.youtube.com/watch?v=3MNVP9-hglc).

    """

    __slots__ = ("_S0",)

    def __init__(self, gtab, S0=None, **kwargs):
        """Implement object initialization."""
        if S0 is None:
            raise ValueError("S0 must be provided")

        self._S0 = S0

    def fit(self, *args, **kwargs):
        """Do nothing."""

    def predict(self, gradient, **kwargs):
        """Return the *b=0* map."""
        return self._S0


class TensorModel:
    """A wrapper of :obj:`dipy.reconst.dti.TensorModel."""

    __slots__ = ("_model", "_S0", "_mask")

    def __init__(self, gtab, S0=None, mask=None, **kwargs):
        """Instantiate the wrapped tensor model."""
        from dipy.reconst.dti import TensorModel

        self._S0 = np.clip(
            S0.astype("float32") / S0.max(),
            a_min=1e-5,
            a_max=1.0,
        )
        self._mask = mask
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ("min_signal", "return_S0_hat", "fit_method", "weighting", "sigma", "jac")
        }
        self._model = TensorModel(gtab, **kwargs)

    def fit(self, data, **kwargs):
        """Clean-up permitted args and kwargs, and call model's fit."""
        self._model = self._model.fit(data, mask=self._mask)

    def predict(self, gradient, step=None, **kwargs):
        """Propagate model parameters and call predict."""
        return self._model.predict(
            _rasb2dipy(gradient),
            S0=self._S0,
            step=step,
        )


class DKIModel:
    """A wrapper of :obj:`dipy.reconst.dki.DiffusionKurtosisModel."""

    __slots__ = ("_model", "_S0", "_mask")

    def __init__(self, gtab, S0=None, mask=None, **kwargs):
        """Instantiate the wrapped tensor model."""
        from dipy.reconst.dki import DiffusionKurtosisModel

        self._S0 = np.clip(
            S0.astype("float32") / S0.max(),
            a_min=1e-5,
            a_max=1.0,
        )
        self._mask = mask
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ("min_signal", "return_S0_hat", "fit_method", "weighting", "sigma", "jac")
        }
        self._model = DiffusionKurtosisModel(gtab, **kwargs)

    def fit(self, data, **kwargs):
        """Clean-up permitted args and kwargs, and call model's fit."""
        self._model = self._model.fit(data, mask=self._mask)

    def predict(self, gradient, step=None, **kwargs):
        """Propagate model parameters and call predict."""
        return self._model.predict(
            _rasb2dipy(gradient),
            S0=self._S0,
            step=step,
        )


def _rasb2dipy(gradient):
    gradient = np.asanyarray(gradient)
    if gradient.ndim == 1:
        if gradient.size != 4:
            raise ValueError("Missing gradient information.")
        gradient = gradient[..., np.newaxis]

    if gradient.shape[0] != 4:
        gradient = gradient.T
    elif gradient.shape == (4, 4):
        print("Warning: make sure gradient information is not transposed!")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        retval = gradient_table(gradient[3, :], gradient[:3, :].T)
    return retval
