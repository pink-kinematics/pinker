***************
Developer notes
***************

This section documents internal functions and other notes shared between
contributors to this project.

Development environment
=======================

Everything runs through `pixi <https://pixi.sh/>`__, which provides the C
compiler and every dependency:

.. code:: bash

    pixi run -e test-py312 test   # run tests
    pixi run build                # compile the C extension
    pixi run dev-lint             # clang-format, mypy, pylint and ruff
    pixi run docs-open            # build this documentation and open it

The test task currently requires specifying which Python version to use by
specifying the corresponding ``test-py3xx`` environment. For the other tasks,
pixi will pick up the appropriate environment automatically.

You can also use the ``example`` task to run examples in the ``examples``
environment. It takes the path to the example as argument:

.. code:: bash

    pixi run example examples/ur3_end_effector_tracking.py

Design guidelines
=================

- Pinker is designed for clarity before performance (except in the C extension
  implemented ``pinker/kinematics/_kinematics_c.c``, whose code is more terse).
- Exceptions raised by the library all derive from a Pinker exception base
  class to avoid abstraction leakage. See this `design decision
  <https://github.com/getparthenon/parthenon/wiki/Design-Decision:-Throw-Custom-Exceptions>`__
  for more details on the rationale behind this choice.
- Task representation strings:
    - We commonly define `__repr__` at the bottom of task Python source files.
    - Only report parameters that have an effect (for instance, the damping
      task does not report its :code:`lm_damping` since its error is always
      zero).
    - Parent-class attributes come after the class's own.

Testing
=======

The test suite covers both the Python library and the C extension for the
kinematics backend, and ``test`` runs the two of them after rebuilding the
extension:

.. code:: bash

    pixi run -e test-py312 test

Cross-validation tests are included in
``tests/kinematics/test_vs_pinocchio.py`` to compare the backend's outputs with
Pinocchio's as a reference implementation, with a precision threshold set to
:math:`10^{-10}` on the same models.

Exceptions
==========

.. automodule:: pinker.exceptions
    :members:
