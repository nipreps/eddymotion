#!/usr/bin/env python
"""A convenience tool for querying emc's version."""
import sys
import os.path as op


def main():
    """Print current dMRIPrep version."""
    sys.path.insert(0, op.abspath('.'))
    from dmriprep.__about__ import __version__
    print(__version__)


if __name__ == '__main__':
    main()
