.. _Examples:

********
Examples
********

A first script
==============

Let us go through ``examples/ur3_end_effector_tracking.py``, where a UR3 arm
tracks a target moving back and forth. We first load the robot model:

.. code:: python

    import pinker

    robot = pinker.load_robot_description("ur3_official_description")

We define the two tasks of this inverse kinematics: track the target with the
end effector, and stay close to a reference posture. The posture task has a
much lower cost, so that it only regularizes the redundancy left by the first
task:

.. code:: python

    from pinker.tasks import FrameTask, PostureTask

    end_effector_task = FrameTask(
        "tool0",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=1.0,  # tuned for this setup
    )
    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )
    tasks = [end_effector_task, posture_task]

Tasks are defined by their cost and by their target. Here we initialize all
targets from the initial configuration of the robot, so that the robot starts
at rest with a zero task error:

.. code:: python

    import pinker
    from pinker.kinematics import custom_configuration

    q_ref = custom_configuration(
        robot.model,
        elbow_joint=1.0,
        shoulder_lift_joint=1.0,
        shoulder_pan_joint=1.0,
    )
    configuration = pinker.Configuration(robot.model, robot.data, q_ref)
    for task in tasks:
        task.set_target_from_configuration(configuration)

We can now run the closed-loop inverse kinematics. Each cycle updates the
target of the end-effector task, computes a velocity that steers the robot
towards all its tasks, and integrates that velocity into the next
configuration:

.. code:: python

    import numpy as np
    from loop_rate_limiters import RateLimiter

    from pinker import solve_ik

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        end_effector_target = end_effector_task.transform_target_to_world
        end_effector_target.translation[1] = 0.5 + 0.1 * np.sin(2.0 * t)
        end_effector_target.translation[2] = 0.2
        velocity = solve_ik(configuration, tasks, dt, solver="daqp")
        configuration.integrate_inplace(velocity, dt)
        rate.sleep()
        t += dt

The configuration is thus the state carried from one cycle to the next. The
full example wraps this loop with a `Viser <https://viser.studio>`__
visualizer:

.. code:: python

    from pinker.visualizer import start_viser_visualizer

    viz = start_viser_visualizer(robot)
    viz.display(configuration.q)  # in the loop, after integration

Running the examples
====================

Examples live in the ``examples/`` directory of the repository and run in the
``examples`` pixi environment, which provides the visualizer and the robot
descriptions:

.. code:: bash

    pixi run -e examples example examples/ur3_end_effector_tracking.py

The command starts a Viser server and opens the visualization in a new browser
tab.

Alternatively, every example carries its dependencies in an inline script
metadata block (`PEP 723 <https://peps.python.org/pep-0723/>`__), so that `uv
<https://docs.astral.sh/uv/>`__ can run it standalone:

.. code:: bash

    uv run examples/ur3_end_effector_tracking.py

The block points ``pinker`` at the repository it lives in, so this works from a
clone (uv compiles the C extension) without installing anything first.

What the examples cover
=======================

.. list-table::
    :widths: 40 60
    :header-rows: 1

    * - Example
      - Illustrates
    * - ``cookie_visualization.py``
      - Displaying a robot description in Viser
    * - ``draco3_reaching.py``
      - Closed kinematic chains with :class:`.JointCouplingTask`
    * - ``g1_com_tracking.py``, ``jvrc_com_tracking.py``
      - Center-of-mass tracking with :class:`.ComTask`
    * - ``gen2_end_effector_tracking.py``,
        ``panda_end_effector_tracking.py``, ``ur3_end_effector_tracking.py``,
        ``ur5_end_effector_tracking.py``
      - End-effector tracking with a posture regularization
    * - ``go2_squat_barrier.py``, ``ur5_position_barrier.py``
      - Control barrier functions (:ref:`Barriers`)
    * - ``jvrc_reaching.py``, ``sigmaban_standing.py``,
        ``upkie_crouching.py``
      - Whole-body inverse kinematics, with the feet in contact
    * - ``piper_inverse_kinematics.py``, ``ur10_inverse_kinematics.py``
      - Iterating differential IK to reach a prescribed end-effector pose
    * - ``stretch_mobile_manipulation.py``, ``stretch_relative_target.py``,
        ``stretch_world_target.py``
      - Mobile manipulation, with world and mobile-base targets
    * - ``upkie_floating_base_velocity_limit.py``
      - Clamping base velocities with :class:`.FloatingBaseVelocityLimit`
    * - ``upkie_rolling.py``
      - Rolling without slipping, with :class:`.RollingTask`
    * - ``ur3_sparse_solver.py``
      - Selecting a sparse QP solver
    * - ``ur3_velocity_smoothing.py``
      - Smoothing velocities with a task gain, a :class:`.DampingTask` and an
        :class:`.AccelerationLimit`
    * - ``z1_joint_velocity_tracking.py``
      - Tracking joint velocities with :class:`.JointVelocityTask`

Examples are named ``<robot>_<task>.py``, after the robot description they load
and what they do with it. Their ``README.md`` has videos of the resulting
motions.
