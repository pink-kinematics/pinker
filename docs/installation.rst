************
Installation
************

Pinker is not published to PyPI or conda-forge yet, so for now it is installed
from source. Its C extension is compiled at installation time, which requires a
C compiler and the NumPy headers; everything else is pure Python.

From source
===========

.. code:: bash

    pip install git+https://github.com/pink-kinematics/pinker.git

The runtime dependencies are `NumPy <https://numpy.org/>`__ and `qpsolvers
<https://github.com/qpsolvers/qpsolvers>`__ (which brings in a QP solver of
your choice). Pinker requires Python 3.10 or later.

Since the library ships no QP solver of its own, install at least one, for
instance:

.. code:: bash

    pip install daqp

Solvers available in your environment are listed in
``qpsolvers.available_solvers``, and selected by name in
:func:`pinker.solve_ik.solve_ik`.

For development
===============

The repository is a `pixi <https://pixi.sh/>`__ project, which provides the
compiler and all dependencies:

.. code:: bash

    git clone https://github.com/pink-kinematics/pinker.git
    cd pinker
    pixi run build

The ``build`` task compiles the C extension and installs the project in
editable mode. It is a dependency of the other tasks, so that for instance:

.. code:: bash

    pixi run test                # build, then run the test suite
    pixi run -e docs docs-build  # build, then build this documentation

Rebuild after any change to ``pinker/kinematics/_kinematics_c.c``.

Robot descriptions
==================

The examples and tests load their robot models from `robot_descriptions
<https://github.com/robot-descriptions/robot_descriptions.py>`__:

.. code:: bash

    pip install robot_descriptions

Descriptions whose URDF is generated from xacro, such as the official
Universal Robots arms, also require `xacrodoc
<https://github.com/adamheins/xacrodoc>`__:

.. code:: bash

    pip install xacrodoc

Descriptions are then loaded by name with :func:`.load_robot_description`:

.. code:: python

    import pinker

    robot = pinker.load_robot_description("ur3_official_description")

Local URDF files are loaded with :func:`.load_robot_urdf`, which does not
require the ``robot_descriptions`` package:

.. code:: python

    robot = pinker.load_robot_urdf("robot.urdf")
