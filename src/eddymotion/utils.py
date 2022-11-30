from pathlib import Path, PurePosixPath
import nibabel as nb
import ants
from ants import ants_image as iio
from ants import lib


def get_lib_fn(string):
    return lib.__dict__[string]


def _ptrstr(pointer):
    """ get string representation of a py::capsule (aka pointer) """
    libfn = get_lib_fn("ptrstr")
    return libfn(pointer)


def antsimage_from_path(val):
    """
    Returns ANTsImage from .nii or .gz path.

    Parameters
    ----------
    val : 
        Value in dictionary of registration arguments

    Returns
    -------
    :obj:`ANTsImage` if val is a string corresponding to a path ending in .nii.gz or .nii
    """
    p = Path(val)
    if p.exists():
        return ants.from_nibabel(nb.load(p))
    else:
        return val


def process_args(args):
    """
    Returns processed registration arguments

    Parameters
    ----------
    args : 
        Dictionary of unprocessed registration arguments

    Returns
    -------
    p_args : 
        Dictionary of processed registration arguments
    """

    # Adapted from _int_antsProcessArguments in ANTsPy utils/process_args.py
    p_args = dict()
    for argname, argval in args.items():
        if "-MULTINAME-" in argname:
            argname = argname[: argname.find("-MULTINAME-")]
        if argval is not None:
            #if len(argname) > 1:
            #    argname = "--%s" % argname
            #else:
            #    argname = "-%s" % argname

            if isinstance(argval, str):
                if Path(argval).exists() and (PurePosixPath(argval).suffix == '.nii.gz' or PurePosixPath(argval).suffix == '.nii'):
                    argval = ants.from_nibabel(nb.load(val))

            if isinstance(argval, iio.ANTsImage):
                p_args[argname] = _ptrstr(argval.pointer)
            elif isinstance(argval, list):
                p = []
                for av in argval:
                    if isinstance(av, iio.ANTsImage):
                        av = _ptrstr(av.pointer)
                    elif str(arg) == "True":
                        av = str(1)
                    elif str(arg) == "False":
                        av = str(0)
                    p.append(av)
                p_args[argname] = p
            else:
                p_args[argname] = str(argval)
    
        return p_args