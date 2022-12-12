"""A factory class that adapts DIPY's dMRI models."""
import warnings
from joblib import Parallel, delayed
import numpy as np
from dipy.core.gradients import gradient_table
from importlib import import_module


def _exec_fit(model, data, chunk=None):
    retval = model.fit(data)
    return retval, chunk


def _exec_predict(model, gradient, chunk=None, **kwargs):
    """Propagate model parameters and call predict."""
    return np.squeeze(model.predict(gradient, S0=kwargs.pop("S0", None))), chunk


class ModelFactory:
    """A factory for instantiating diffusion models."""

    @staticmethod
    def init(gtab, model="DTI", **kwargs):
        """
        Instatiate a diffusion model.

        Parameters
        ----------
        gtab : :obj:`numpy.ndarray`
            An array representing the gradient table in RAS+B format.
        model : :obj:`str`
            Diffusion model.
            Options: ``"DTI"``, ``"DKI"``, ``"S0"``, ``"AverageDW"``

        Return
        ------
        model : :obj:`~dipy.reconst.ReconstModel`
            An model object compliant with DIPY's interface.

        """

        if model.lower() in ("s0", "b0"):
            return TrivialB0Model(gtab=gtab, S0=kwargs.pop("S0"))

        if model.lower() in ("avg", "average", "mean"):
            return AverageDWModel(gtab=gtab, **kwargs)

        # Generate a GradientTable object for DIPY
        if model.lower() in ("dti", "dki"):
            Model = globals()[f"{model.upper()}Model"]
            return Model(gtab, **kwargs)

        raise NotImplementedError(f"Unsupported model <{model}>.")


class BaseModel:
    """
    Defines the interface and default methods.

    Implements the interface of :obj:`dipy.reconst.base.ReconstModel`.
    Instead of inheriting from the abstract base, this implementation
    follows type adaptation principles, as it is easier to maintain
    and to read (see https://www.youtube.com/watch?v=3MNVP9-hglc).

    """

    __slots__ = ("_model", "_mask", "_S0", "_b_max", "_models", "_datashape", "_n_models")
    _modelargs = tuple()

    def __init__(self, gtab, S0=None, mask=None, b_max=None, **kwargs):
        """Base initialization."""

        # Setup brain mask
        self._mask = mask
        if "mask" in kwargs:
            self._mask = kwargs.pop("mask")
        if mask is None and S0 is not None:
            self._mask = self._S0 > np.percentile(self._S0, 35)

        # Setup B0 map
        if "S0" in kwargs:
            S0 = kwargs.pop("S0")
        if S0 is not None:
            self._S0 = np.clip(
                S0.astype("float32") / S0.max(),
                a_min=1e-5,
                a_max=1.0,
            )
            # Select voxels within mask or just unravel 3D if no mask
            self._S0 = (
                np.ma.masked_array(self._S0, mask=np.broadcast_to(self._mask, self._S0.shape)).data
                if self._mask is not None
                else self._S0.reshape(-1, self._S0.shape[-1])
            )

        # Cap b-values, if requested
        if "b_max" in kwargs:
            b_max = kwargs.pop("b_max")
        if b_max and b_max > 1000:
            # Saturate b-values at b_max, since signal stops dropping
            gtab[-1, gtab[-1] > b_max] = b_max
            # A possibly good alternative is completely remove very high b-values
            # bval_mask = gtab[-1] < b_max
            # data = data[..., bval_mask]
            # gtab = gtab[:, bval_mask]
            self._b_max = b_max
        else:
            self._b_max = None

        kwargs = {k: v for k, v in kwargs.items() if k in self._modelargs}

        model_str = getattr(self, "_model_class", None)
        if not model_str:
            raise TypeError("No model defined")

        module_name, class_name = model_str.rsplit(".", 1)
        self._model = getattr(import_module(module_name), class_name)(_rasb2dipy(gtab), **kwargs)

    def fit(self, data, n_jobs=None, **kwargs):
        """Fit the model chunk-by-chunk asynchronously"""
        n_jobs = n_jobs or 1

        self._datashape = data.shape

        # Add fourth axis to mask if missing
        mask = (
            self._mask[..., None] if self._mask is not None and self._mask.ndim == 3 else self._mask
        )
        # Select voxels within mask or just unravel 3D if no mask
        data = (
            np.ma.masked_array(data, mask=np.broadcast_to(mask, data.shape)).data
            if mask is not None
            else data.reshape(-1, data.shape[-1])
        )

        # One single CPU - linear execution (full model)
        if n_jobs == 1:
            self._model, _ = _exec_fit(self._model, data)
            return

        self._models = [None] * n_jobs

        # Parallelize process with joblib
        # Split data into chunks of group of slices
        with Parallel(n_jobs=n_jobs) as executor:
            results = executor(
                delayed(_exec_fit)(self._model, dchunk, i)
                for i, dchunk in enumerate(np.array_split(data, n_jobs))
            )
        if results:
            for submodel, index in results:
                self._models[index] = submodel
        else:
            raise RuntimeError("No results from parallel execution across data chunks.")

        self._model = None  # Preempt further actions on the model

        self._n_models = len(self._models) if self._model is None and self._models else 1

    def predict(self, gradient, **kwargs):
        """Predict asynchronously chunk-by-chunk the diffusion signal."""
        if self._b_max is not None:
            gradient[-1] = min(gradient[-1], self._b_max)

        gradient = _rasb2dipy(gradient)

        if self._n_models == 1:
            S0 = self._S0
            predicted, _ = _exec_predict(self._model, gradient, S0=S0, **kwargs)
        else:
            S0 = (
                np.array_split(self._S0, self._n_models)
                if self._S0 is not None
                else [None] * self._n_models
            )

            predicted = [None] * self._n_models

            # Parallelize process with joblib
            with Parallel(n_jobs=self._n_models) as executor:
                results = executor(
                    delayed(_exec_predict)(model, gradient, S0=S0[i], chunk=i, **kwargs)
                    for i, model in enumerate(self._models)
                )

            if results:
                predicted = np.vstack([r[0] for r in results])
                if self._mask is not None:
                    retval = np.zeros_like(self._mask, dtype="float32")
                    if self._mask.ndim == 3:
                        mask = self._mask.reshape(-1)
                        retval = retval.reshape(-1)
                    retval[mask] = predicted.reshape(-1)
                    retval = retval.reshape(self._datashape[:-1])
                else:
                    retval = predicted.reshape(self._datashape[:-1])
                return retval

            else:
                raise RuntimeError("No results from parallel execution across data chunks.")


class TrivialB0Model:
    """A trivial model that returns a *b=0* map always."""

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


class AverageDWModel:
    """A trivial model that returns an average map."""

    __slots__ = ("_data", "_th_low", "_th_high", "_bias", "_stat")

    def __init__(self, gtab, **kwargs):
        r"""
        Implement object initialization.

        Parameters
        ----------
        gtab : :obj:`~numpy.ndarray`
            An :math:`N \times 4` table, where rows (*N*) are diffusion gradients and
            columns are b-vector components and corresponding b-value, respectively.
        th_low : :obj:`~numbers.Number`
            A lower bound for the b-value corresponding to the diffusion weighted images
            that will be averaged.
        th_high : :obj:`~numbers.Number`
            An upper bound for the b-value corresponding to the diffusion weighted images
            that will be averaged.
        bias : :obj:`bool`
            Whether the overall distribution of each diffusion weighted image will be
            standardized and centered around the global 75th percentile.
        stat : :obj:`str`
            Whether the summary statistic to apply is ``"mean"`` or ``"median"``.

        """
        self._th_low = kwargs.get("th_low", 50)
        self._th_high = kwargs.get("th_high", 10000)
        self._bias = kwargs.get("bias", True)
        self._stat = kwargs.get("stat", "median")

    def fit(self, data, **kwargs):
        """Calculate the average."""
        gtab = kwargs.pop("gtab", None)
        # Select the interval of b-values for which DWIs will be averaged
        b_mask = (
            ((gtab[3] >= self._th_low) & (gtab[3] <= self._th_high))
            if gtab is not None
            else np.ones((data.shape[-1],), dtype=bool)
        )
        shells = data[..., b_mask]

        # Regress out global signal differences
        if self._bias:
            centers = np.median(shells, axis=(0, 1, 2))
            reference = np.percentile(centers[centers >= 1.0], 75)
            centers[centers < 1.0] = reference
            drift = reference / centers
            shells = shells * drift

        # Select the summary statistic
        avg_func = np.median if self._stat == "median" else np.mean
        # Calculate the average
        self._data = avg_func(shells, axis=-1)

    def predict(self, gradient, **kwargs):
        """Return the average map."""
        return self._data


class DTIModel(BaseModel):
    """A wrapper of :obj:`dipy.reconst.dti.TensorModel`."""

    _modelargs = (
        "min_signal",
        "return_S0_hat",
        "fit_method",
        "weighting",
        "sigma",
        "jac",
    )
    _model_class = "dipy.reconst.dti.TensorModel"


class DKIModel(BaseModel):
    """A wrapper of :obj:`dipy.reconst.dki.DiffusionKurtosisModel`."""

    _modelargs = DTIModel._modelargs
    _model_class = "dipy.reconst.dki.DiffusionKurtosisModel"


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
