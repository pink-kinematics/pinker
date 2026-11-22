.. title:: Table of Contents

######
Pinker
######

**P**\ ython **in**\ verse **k**\ inematics for **e**\ mbedded **r**\ obotics.

Inverse kinematics in Pinker is defined by weighted :ref:`tasks <Tasks>`,
:ref:`limits <Limits>` and :ref:`barriers <Barriers>`. Once a robot is loaded,
its current geometric state is held in its :ref:`configuration
<Configuration>`. Given a configuration, tasks and a time step,
:func:`pinker.solve_ik.solve_ik` computes joint velocities that steer the model
towards fulfilling all tasks at best:

.. code:: python

    velocity = solve_ik(configuration, tasks, dt, solver="daqp")
    configuration.integrate_inplace(velocity, dt)

To get started, :doc:`install Pinker <installation>` and a QP solver of your
choice, then follow the first script in :ref:`Examples`. The
:doc:`introduction` sets out notations and introduces the two main concepts of
robot configuration and task used in the library.

.. toctree::
    :caption: Getting started
    :maxdepth: 1

    installation.rst
    introduction.rst
    examples.rst

.. toctree::
    :caption: API documentation
    :maxdepth: 1

    tasks.rst
    limits.rst
    barriers.rst
    inverse-kinematics.rst
    kinematics.rst
    visualization.rst
    developer-notes.rst
    references.rst

Index
~~~~~

:ref:`genindex`
