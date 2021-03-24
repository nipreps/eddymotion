"""A factory class that adapts DIPY's dMRI models."""
import attr
from dipy import reconst as dr
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
            return TrivialB0Model()

        # Generate a GradientTable object for DIPY
        gtab = gradient_table(gtab)
        param = {}

        if model.lower().startswith("3dshore"):
            param = {
                "radial_order": 6,
                "zeta": 700,
                "lambdaN": 1e-8,
                "lambdaL": 1e-8,
            }
            Model = dr.shore.ShoreModel

        elif model.lower().startswith("sfm"):
            from emc.utils.model import SFM4HMC, ExponentialIsotropicModel

            param = {
                "isotropic": ExponentialIsotropicModel,
            }
            Model = SFM4HMC

        elif model.lower().startswith("tensor"):
            Model = dr.dti.TensorModel

        else:
            raise NotImplementedError(f"Unsupported model <{model}>.")

        param.update(kwargs)
        return Model(gtab, **param)


@attr.s(slots=True, frozen=True)
class TrivialB0Model:
    """
    A trivial model that returns a *b=0* map always.

    Imlpements the interface of :obj:`dipy.reconst.base.ReconstModel`.
    Instead of inheriting from the abstract base, this implementation
    follows type adaptation principles, as it is easier to maintain
    and to read (see https://www.youtube.com/watch?v=3MNVP9-hglc).

    """

    def fit(self, *args, **kwargs):
        """Do nothing."""

    def predict(self, *args, **kwargs):
        """Return the *b=0* map."""
        return kwargs["S0"]
