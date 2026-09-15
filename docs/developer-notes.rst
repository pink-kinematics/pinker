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

    pixi run build              # compile the C extension, install in editable mode
    pixi run test               # build, then run the test suite
    pixi run -e lint lint       # ruff check and ruff format --check
    pixi run -e docs docs-open  # build this documentation and open it

The tasks that need the library depend on ``build``, so the C extension is
rebuilt before they run. Pass ``-e docs`` to the documentation tasks: without
an explicit environment, pixi runs their ``build`` dependency in the heavier
default environment.

Design guidelines
=================

* Pinker is designed for clarity before performance, except in the backend
  where the trade-off is reversed: ``pinker/kinematics/_kinematics_c.c`` is
  where the library is allowed to be terse.
* Everything above the backend is Pink's code with the backend import swapped.
  Diffs with upstream Pink are kept minimal so that future Pink releases can be
  re-ported.
* Exceptions raised by the library all derive from a Pinker exception base
  class to avoid abstraction leakage. See this `design decision
  <https://github.com/getparthenon/parthenon/wiki/Design-Decision:-Throw-Custom-Exceptions>`__
  for more details on the rationale behind this choice.
* Task representation strings:
    * Only report parameters that have an effect (for instance, the damping
      task does not report its :code:`lm_damping` since its error is always
      zero).
    * Parent-class attributes come after the class's own.
    * A *de facto* practice has emerged where we define `__repr__` at the
      bottom of task Python source files.

.. _Differences:

Testing
=======

The test suite covers both the Python library and the C extension for the
kinematics backend:

.. code:: bash

    pixi run test             # full suite, after rebuilding the C extension
    pixi run test-kinematics  # backend tests only, including cross-validation

Cross-validation tests are included in
``tests/kinematics/test_vs_pinocchio.py`` to compare the backend's outputs with
Pinocchio's as a reference implementation, with a precision threshold set to
:math:`10^{-10}` on the same models.

Exceptions
==========

.. automodule:: pinker.exceptions
    :members:
