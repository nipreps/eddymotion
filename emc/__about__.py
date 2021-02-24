"""Base module variables."""
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__packagename__ = 'emc'
__copyright__ = 'Copyright 2021, The EddyMotionCorrection developers'
__url__ = 'https://github.com/dPys/EddyMotionCorrection'

DOWNLOAD_URL = (
    'https://github.com/dPys/{name}/archive/{ver}.tar.gz'.format(
        name=__packagename__, ver=__version__))

