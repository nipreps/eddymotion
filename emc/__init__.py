"""Top-level package for emc."""
from emc._version import __version__

__packagename__ = "emc"
__copyright__ = "Copyright 2021, The EddyMotionCorrection developers"
__url__ = "https://github.com/nipreps/EddyMotionCorrection"

DOWNLOAD_URL = (
    f"https://github.com/nipreps/{__packagename__}/archive/{__version__}.tar.gz"
)


__all__ = [
    "__version__",
    "__copyright__",
    "__packagename__",
]
