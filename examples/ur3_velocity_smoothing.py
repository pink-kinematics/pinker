# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["daqp", "loop-rate-limiters", "matplotlib", "pinker",
# "pycollada", "qpsolvers", "robot_descriptions >=3.1.0", "trimesh", "viser",
# "xacrodoc"]
#
# [tool.uv.sources]
# pinker = { path = "..", editable = true }
# ///

"""UR3 arm tracking a target, first without then with velocity smoothing."""

import matplotlib.pyplot as plt
import numpy as np
import qpsolvers
import viser.transforms as vtf
from loop_rate_limiters import RateLimiter

import pinker
from pinker import solve_ik
from pinker.kinematics import custom_configuration
from pinker.limits import AccelerationLimit
from pinker.tasks import DampingTask, FrameTask, PostureTask
from pinker.visualizer import start_viser_visualizer

NB_STEPS = 3000  # number of steps to run the example for

if __name__ == "__main__":
    robot = pinker.load_robot_description("ur3_official_description")

    viz = start_viser_visualizer(robot)
    viewer = viz.viewer
    end_effector_target_frame = viewer.scene.add_frame(
        "/end_effector_target", axes_length=0.1, axes_radius=0.005
    )
    end_effector_frame = viewer.scene.add_frame(
        "/end_effector", axes_length=0.1, axes_radius=0.005
    )

    # Define inverse kinematics tasks and limits
    end_effector_task = FrameTask(
        "tool0",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=1.0,  # tuned for this setup
    )
    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )
    damping_task = DampingTask(
        cost=1e-1,  # [cost] * [s] / [rad]
    )
    acceleration_limit = AccelerationLimit(
        robot.model,
        np.full(
            robot.model.nv,
            20.0,  # [rad] / [s]^2
        ),
    )

    # Initial configuration and task setup
    q_ref = custom_configuration(
        robot.model,
        shoulder_lift_joint=1.0,
        shoulder_pan_joint=1.0,
        elbow_joint=1.0,
    )
    configuration = pinker.Configuration(robot.model, robot.data, q_ref)
    end_effector_task.set_target_from_configuration(configuration)
    posture_task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    configurations, velocities, times = [], [], []
    t = 0.0  # [s]
    max_ever = -1.0
    for step in range(NB_STEPS):
        # Update task targets
        end_effector_target = end_effector_task.transform_target_to_world
        end_effector_target.translation[1] = 0.5 + 0.1 * np.sin(2.0 * t)
        end_effector_target.translation[2] = 0.2

        # Update visualization frames
        _T = end_effector_target.toarray()
        end_effector_target_frame.position = _T[:3, 3]
        end_effector_target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
        _T = configuration.get_transform_frame_to_world(
            end_effector_task.frame
        ).toarray()
        end_effector_frame.position = _T[:3, 3]
        end_effector_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz

        if step < NB_STEPS // 2:
            # First half: no velocity smoothing
            end_effector_task.gain = 1.0
            tasks = (end_effector_task, posture_task)
            limits = configuration.default_limits
        else:  # step >= NB_STEPS // 2
            # Second half: velocity smoothing by:
            # 1. Reducing the task gain
            # 2. Switching from a posture to a damping task
            # 3. Adding an acceleration limit
            end_effector_task.gain = 0.4
            tasks = (end_effector_task, damping_task)
            limits = [*configuration.default_limits, acceleration_limit]

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(
            configuration, tasks, dt, solver=solver, limits=limits
        )
        Delta_q = velocity * dt
        configuration.integrate_inplace(velocity, dt)
        acceleration_limit.set_last_integration(velocity, dt)

        # Append plotting data to lists
        configurations.append(configuration.q)
        times.append(t)
        velocities.append(velocity)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt

    # Plot velocities and configurations over time
    velocities = np.array(velocities)
    configurations = np.array(configurations)
    plt.figure()
    plt.plot(times, velocities)
    plt.plot(
        [times[NB_STEPS // 2]] * 2,
        [velocities.min(), velocities.max()],
        "k--",
    )
    plt.title("Joint velocities (posture then damping task)")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint velocity [rad/s]")
    plt.figure()
    plt.plot(times, configurations)
    plt.plot(
        [times[NB_STEPS // 2]] * 2,
        [configurations.min(), configurations.max()],
        "k--",
    )
    plt.title("Configuration (posture then damping task)")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint position [rad]")
    plt.show()
