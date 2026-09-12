:github_url: https://github.com/pink-kinematics/pinker/tree/main/doc/installation.rst

************
Installation
************

From Conda
==========

For best performance we recommended installing Pinker from Conda:

.. code:: bash

    conda install -c conda-forge pinker

From PyPI
=========

Installation from the Python Package Index should work via:

.. code:: bash

    pip install pinker

From source
===========

Pinker's C extension is compiled at installation time, which requires a C compiler and the NumPy headers; everything else is pure Python.

.. code:: bash

    pip install git+https://github.com/pink-kinematics/pinker.git
