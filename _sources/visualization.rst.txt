.. _Visualization:

*************
Visualization
*************

Pinker displays robots with `Viser <https://viser.studio>`__, which serves the
scene to a Web browser and handles user inputs. You can start a visualizer from
a robot wrapper, then display configurations as the inverse kinematics unrolls:

.. code:: python

    from pinker.visualizer import start_viser_visualizer

    viz = start_viser_visualizer(robot)
    viz.display(configuration.q)

Visuals come from the geometry model that
:func:`~pinker.kinematics.urdf.build_geom_from_urdf` reads from the URDF: see
:ref:`Kinematics` for the geometry types the parser supports.

.. automodule:: pinker.visualizer
    :members:
