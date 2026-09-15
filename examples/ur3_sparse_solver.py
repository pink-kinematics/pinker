# SPDX-License-Identifier: Apache-2.0
#
# /// script
# dependencies = ["clarabel", "loop-rate-limiters", "meshcat", "pinker",
# "qpsolvers", "robot_descriptions", "xacrodoc"]
# ///

"""Arm tracking a circular end-effector motion, use a sparse QP solver."""

import warnings

import numpy as np
import qpsolvers
import viser.transforms as vtf
from loop_rate_limiters import RateLimiter

import pinker
from pinker import solve_ik
from pinker.tasks import FrameTask, PostureTask
from pinker.visualizer import start_viser_visualizer

# Target circle parameters
CENTER = np.array([0.3, 0.0, 0.5])  # m, in the world frame
RADIUS = 0.15  # m


if __name__ == "__main__":
    robot = pinker.load_robot_description("ur3_official_description")

    # Initialize visualization
    viz = start_viser_visualizer(robot)
    viewer = viz.viewer
    target_frame = viewer.scene.add_frame(
        "/target", axes_length=0.1, axes_radius=0.005
    )
    end_effector_frame = viewer.scene.add_frame(
        "/end_effector", axes_length=0.1, axes_radius=0.005
    )

    # Define differential IK tasks
    end_effector_task = FrameTask(
        "tool0",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=0.1,  # [cost] / [rad]
    )
    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )
    tasks = [end_effector_task, posture_task]

    # Initialize configuration
    configuration = pinker.Configuration(robot.model, robot.data, robot.q0)
    for task in tasks:
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    # Select a sparse QP solver
    if not qpsolvers.sparse_solvers:
        raise Exception(
            "this example is for QP solvers with a sparse-matrix API, "
            "yet qpsolvers did not find any. "
            "You can install e.g. Clarabel, OSQP or SCS, then try again."
        )
    sparse_solver = qpsolvers.sparse_solvers[0]

    # Suppress sparse matrix conversion warnings
    warnings.filterwarnings(
        "ignore",
        category=qpsolvers.warnings.SparseConversionWarning,
    )

    # Control loop parameters
    rate = RateLimiter(frequency=100.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]

    while True:
        # Update target to move in a circle
        target_pos = CENTER + RADIUS * np.array([np.cos(t), np.sin(t), 0.1])
        end_effector_task.transform_target_to_world.translation = target_pos

        # Update visualization frames
        _T = np.asarray(end_effector_task.transform_target_to_world.np)
        target_frame.position = _T[:3, 3]
        target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
        _T = np.asarray(
            configuration.get_transform_frame_to_world(
                end_effector_task.frame
            ).np
        )
        end_effector_frame.position = _T[:3, 3]
        end_effector_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz

        # Solve inverse kinematics
        velocity = solve_ik(configuration, tasks, dt, solver=sparse_solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at a fixed framerate
        viz.display(configuration.q)
        rate.sleep()
        t += dt
