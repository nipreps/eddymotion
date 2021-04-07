"""Base module variables."""
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__packagename__ = 'emc'
__copyright__ = 'Copyright 2021, The EddyMotionCorrection developers'
__url__ = 'https://github.com/nipreps/EddyMotionCorrection'

DOWNLOAD_URL = f'https://github.com/nipreps/{__packagename__}/archive/{__version__}.tar.gz'
