************
Installation
************

Pinker is distributed as a source package: its C extension is compiled at
installation time, while everything else in the library is pure Python.

From PyPI
=========

You can install Pinker directly from PyPI, for instance using pip:

.. code:: bash

    pip install pinker

The main runtime dependencies are `NumPy <https://numpy.org/>`__ and `qpsolvers
<https://github.com/qpsolvers/qpsolvers>`__ (which brings in a QP solver of
your choice). Since the library ships no QP solver of its own, you should
install at least one, for instance:

.. code:: bash

    pip install daqp

Solvers available in your environment will be listed in
``qpsolvers.available_solvers``. You can select the backend QP solver by name
when calling :func:`pinker.solve_ik.solve_ik`.

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
