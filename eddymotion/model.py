"""A factory class that adapts DIPY's dMRI models."""
import warnings
import concurrent.futures
import asyncio
import numpy as np
from dipy.core.gradients import gradient_table


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
            Options: ``"3DShore"``, ``"SFM"``, ``"DTI"``, ``"DKI"``, ``"S0"``

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
            from eddymotion.utils.model import (
                SFM4HMC as Model,
                ExponentialIsotropicModel,
            )

            param = {
                "isotropic": ExponentialIsotropicModel,
            }

        elif model.lower() in ("dti", "dki"):
            Model = DTIModel if model.lower() == "dti" else DKIModel

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


class DTIModel:
    """A wrapper of :obj:`dipy.reconst.dti.TensorModel."""

    __slots__ = (
        "_model",
        "_S0",
        "_mask",
        "_nb_threads",
        "_mask_chunks",
        "_model_chunks",
    )

    def __init__(self, gtab, S0=None, mask=None, nb_threads=1, **kwargs):
        """Instantiate the wrapped tensor model."""
        from dipy.reconst.dti import TensorModel as DipyTensorModel

        self._nb_threads = nb_threads
        self._S0 = None
        if S0 is not None:
            self._S0 = np.clip(
                S0.astype("float32") / S0.max(),
                a_min=1e-5,
                a_max=1.0,
            )
        self._mask = mask
        if mask is None and S0 is not None:
            self._mask = self._S0 > np.percentile(self._S0, 35)

        if self._mask is not None:
            self._S0 = self._S0[self._mask.astype(bool)]

            # Create the mask chunks
            self._mask_chunks = np.split(self._mask, self._nb_threads, axis=2)

        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "min_signal",
                "return_S0_hat",
                "fit_method",
                "weighting",
                "sigma",
                "jac",
            )
        }
        self._model = DipyTensorModel(gtab, **kwargs)
        # Create a TensorModel for each chunk
        self._model_chunks = [
            DipyTensorModel(gtab, **kwargs) for _ in range(nb_threads)
        ]

    def fit_chunk(self, data_chunk, index, **kwargs):
        """Clean-up permitted args and kwargs, and call model's fit."""
        self._model_chunks[index] = self._model_chunks[index].fit(
            data_chunk, mask=self._mask_chunks[index]
        )

    async def run_fit_async(self, data, executor):
        """Run the fit asynchronously chunk-by-chunk."""
        print("starting fit")
        print("creating data chunks (group of slices)")
        data_chunks = np.split(data, self._nb_threads, axis=2)

        print("creating executor tasks")
        loop = asyncio.get_event_loop()
        fit_tasks = [
            loop.run_in_executor(executor, self.fit_chunk, data_chunks[i], i)
            for i in range(self._nb_threads)
        ]
        print("waiting for executor tasks")
        results = await asyncio.gather(fit_tasks)
        print(f"results: {results}")

        print("exiting")

    def fit_async(self, data, **kwargs):
        """Run the future :method:`self.run_fit_async` in an asyncio event loop."""
        # Create a limited thread pool.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._nb_threads,
        )

        event_loop = asyncio.get_event_loop()
        try:
            event_loop.run_until_complete(
                self.run_fit_async(data=data, executor=executor)
            )
        finally:
            event_loop.close()

    def fit(self, data, **kwargs):
        """Call model's fit."""
        self._model = self._model.fit(data[self._mask, ...])

    def predict_chunk(self, gradient_chunk, index, step=None):
        """Propagate model parameters and call predict for chunk."""
        return self._model_chunks[index].predict(
            _rasb2dipy(gradient_chunk),
            S0=self._S0,
            step=step,
        )

    async def run_predict_async(self, gradient, executor, step=None):
        """Run the prediction asynchronously chunk-by-chunk."""
        print("starting predict")
        print("creating gradient chunks (group of slices)")
        gradient_chunks = np.split(gradient, self._nb_threads, axis=2)

        print("creating executor tasks")
        loop = asyncio.get_event_loop()
        predict_tasks = [
            loop.run_in_executor(
                executor, self.predict_chunk, gradient_chunks[i], i, step
            )
            for i in range(self._nb_threads)
        ]
        print("waiting for executor tasks")
        results = await asyncio.gather(predict_tasks)
        print(f"results: {results}")

        print("exiting")

    def predict_async(self, gradient, step=None, **kwargs):
        """Run the future :method:`self.run_predict_async` in an asyncio event loop."""
        # Create a limited thread pool.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._nb_threads,
        )

        event_loop = asyncio.get_event_loop()
        try:
            event_loop.run_until_complete(
                self.run_predict_async(gradient=gradient, step=step, executor=executor)
            )
        finally:
            event_loop.close()

    def predict(self, gradient, step=None, **kwargs):
        """Propagate model parameters and call predict."""
        predicted = np.squeeze(
            self._model.predict(
                _rasb2dipy(gradient),
                S0=self._S0,
                step=step,
            )
        )
        if predicted.ndim == 3:
            return predicted

        retval = np.zeros_like(self._mask, dtype="float32")
        retval[self._mask, ...] = predicted
        return retval


class DKIModel:
    """A wrapper of :obj:`dipy.reconst.dki.DiffusionKurtosisModel."""

    __slots__ = ("_model", "_S0", "_mask")

    def __init__(self, gtab, S0=None, mask=None, **kwargs):
        """Instantiate the wrapped tensor model."""
        from dipy.reconst.dki import DiffusionKurtosisModel

        self._S0 = None
        if S0 is not None:
            self._S0 = np.clip(
                S0.astype("float32") / S0.max(),
                a_min=1e-5,
                a_max=1.0,
            )
        self._mask = mask
        if mask is None and S0 is not None:
            self._mask = self._S0 > np.percentile(self._S0, 35)

        if self._mask is not None:
            self._S0 = self._S0[self._mask.astype(bool)]

        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "min_signal",
                "return_S0_hat",
                "fit_method",
                "weighting",
                "sigma",
                "jac",
            )
        }
        self._model = DiffusionKurtosisModel(gtab, **kwargs)

    def fit(self, data, **kwargs):
        """Clean-up permitted args and kwargs, and call model's fit."""
        self._model = self._model.fit(data[self._mask, ...])

    def predict(self, gradient, **kwargs):
        """Propagate model parameters and call predict."""
        predicted = np.squeeze(
            self._model.predict(
                _rasb2dipy(gradient),
                S0=self._S0,
            )
        )
        if predicted.ndim == 3:
            return predicted

        retval = np.zeros_like(self._mask, dtype="float32")
        retval[self._mask, ...] = predicted
        return retval


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
