For developers
==============
Contributing
------------
*Eddymotion* is a project of the *NiPreps Community*, `which specifies the contributing guidelines <https://www.nipreps.org/community/>`__.
Before delving into the code, please make sure you have read all the guidelines offered online.

Documentation
-------------
Documentation sources are found under the ``docs/`` folder, and builds are archived in the `gh-pages <https://github.com/nipreps/eddymotion/tree/gh-pages>`__ branch of the repository.
With GitHub Pages, the documentation is posted under https://www.nipreps.org/eddymotion.
We maintain versioned documentation, by storing git tags under ``<major>.<minor>/`` folders, i.e., we do not archive every patch release, but only every minor release.
In other words, folder ``0.1/`` of the documentation tree contains the documents for the latest release within the *0.1.x* series.
With every commit (or merge commit) to ``main``, the *development* version of the documentation under the folder ``main/`` is updated too.
The ``gh-pages`` branch is automatically maintained with `a GitHub Action <https://github.com/nipreps/eddymotion/blob/main/.github/workflows/docs-build-update.yml>`__.
Please, do not commit manually to ``gh-pages``.

To build the documentation locally, you first need to make sure that ``setuptools_scm[toml] >= 6.2`` is installed in your environment and then::

  $ cd <eddymotion-repository>/
  $ python -m setuptools_scm  # This will generate ``src/eddymotion/_version.py``
  $ make -C docs/ html

Library API (application program interface)
-------------------------------------------
Information on specific functions, classes, and methods.

.. toctree::
   :glob:

   api/eddymotion.cli
   api/eddymotion.data
   api/eddymotion.data.dmri
   api/eddymotion.estimator
   api/eddymotion.exceptions
   api/eddymotion.math
   api/eddymotion.model
   api/eddymotion.utils
   api/eddymotion.viz
