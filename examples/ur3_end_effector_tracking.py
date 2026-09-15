# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["daqp", "loop-rate-limiters", "pinker", "pycollada",
# "qpsolvers", "robot_descriptions >=3.1.0", "trimesh", "viser", "xacrodoc"]
#
# [tool.uv.sources]
# pinker = { path = "..", editable = true }
# ///

"""Universal Robots UR3 arm tracking a moving target."""

import numpy as np
import qpsolvers
import viser.transforms as vtf
from loop_rate_limiters import RateLimiter

import pinker
from pinker import solve_ik
from pinker.tasks import FrameTask, PostureTask
from pinker.kinematics import custom_configuration
from pinker.visualizer import start_viser_visualizer

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

    q_ref = custom_configuration(
        robot.model,
        shoulder_lift_joint=1.0,
        shoulder_pan_joint=1.0,
        elbow_joint=1.0,
    )
    configuration = pinker.Configuration(robot.model, robot.data, q_ref)
    for task in tasks:
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update task targets
        end_effector_target = end_effector_task.transform_target_to_world
        end_effector_target.translation[1] = 0.5 + 0.1 * np.sin(2.0 * t)
        end_effector_target.translation[2] = 0.2

        # Update visualization frames
        _T = np.asarray(end_effector_target.np)
        end_effector_target_frame.position = _T[:3, 3]
        end_effector_target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
        _T = np.asarray(
            configuration.get_transform_frame_to_world(
                end_effector_task.frame
            ).np
        )
        end_effector_frame.position = _T[:3, 3]
        end_effector_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
