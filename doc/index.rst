:github_url: https://github.com/pink-kinematics/pinker/tree/main/doc/index.rst

.. title:: Table of Contents

######
Pinker
######

**P**\ ython **in**\ verse **k**\ inematics for **e**\ mbedded **r**\ obots.

.. image:: https://user-images.githubusercontent.com/1189580/192318997-ed7574c3-8238-451d-9548-a769d46ec03b.png
   :alt: Banner for Pinker

Inverse kinematics in Pinker is defined by weighted :ref:`tasks <Tasks>` and :ref:`limits <Limits>`. The library adds a :ref:`configuration <Configuration>` type, a configuration being a robot model and data to which forward kinematics have been applied. Given a configuration, tasks and a time step, :func:`pinker.solve_ik.solve_ik` computes joint velocities that steer the model towards fulfilling all tasks at best.

.. toctree::
    :maxdepth: 1

    installation.rst
    introduction.rst
    tasks.rst
    limits.rst
    barriers.rst
    inverse-kinematics.rst
    kinematics.rst
    visualization.rst
    developer-notes.rst
    references.rst

