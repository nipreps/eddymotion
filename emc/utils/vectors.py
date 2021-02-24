"""
Vector prediction tools
"""
from pathlib import Path
from itertools import permutations
import numpy as np

B0_THRESHOLD = 50
BVEC_NORM_EPSILON = 0.1


def _nonoverlapping_qspace_samples(
    prediction_bval, prediction_bvec, all_bvals, all_bvecs, cutoff
):
    """Ensure that none of the training samples are too close to the sample to
     predict.
    """
    min_bval = min(min(all_bvals), prediction_bval)
    all_qvals = np.sqrt(all_bvals - min_bval)
    prediction_qval = np.sqrt(prediction_bval - min_bval)

    # Convert q values to percent of maximum qval
    max_qval = max(max(all_qvals), prediction_qval)
    all_qvals_scaled = all_qvals / max_qval * 100
    scaled_qvecs = all_bvecs * all_qvals_scaled[:, np.newaxis]
    scaled_prediction_qvec = prediction_bvec * \
                             (prediction_qval / max_qval * 100)

    # Calculate the distance between the sampled qvecs and the prediction qvec
    ok_samples = (
        np.linalg.norm(scaled_qvecs - scaled_prediction_qvec, axis=1
                       ) > cutoff
    ) * (np.linalg.norm(scaled_qvecs + scaled_prediction_qvec, axis=1
                        ) > cutoff)

    return ok_samples


def _rasb_to_bvec_list(in_rasb):
    """
    Create a list of b-vectors from a rasb gradient table.

    Parameters
    ----------
    in_rasb : str or os.pathlike
        File path to a RAS-B gradient table.
    """
    import numpy as np

    ras_b_mat = np.genfromtxt(in_rasb, delimiter="\t")
    bvec = [vec for vec in ras_b_mat[:, 0:3] if not np.isclose(all(vec), 0)]
    return list(bvec)


def _rasb_to_bval_floats(in_rasb):
    """
    Create a list of b-values from a rasb gradient table.

    Parameters
    ----------
    in_rasb : str or os.pathlike
        File path to a RAS-B gradient table.
    """
    import numpy as np

    ras_b_mat = np.genfromtxt(in_rasb, delimiter="\t")
    return [float(bval) for bval in ras_b_mat[:, 3] if bval > 0]
