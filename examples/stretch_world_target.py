# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["clarabel", "loop-rate-limiters", "pinker", "pycollada",
# "qpsolvers", "robot_descriptions >=3.1.0", "trimesh", "viser"]
#
# [tool.uv.sources]
# pinker = { path = "..", editable = true }
# ///

"""Move a Stretch RE1 with a fixed fingertip target around the origin."""

import numpy as np
import qpsolvers
import viser.transforms as vtf
from loop_rate_limiters import RateLimiter

import pinker
from pinker import PinkerError, solve_ik
from pinker import kinematics as kin
from pinker.tasks import FrameTask
from pinker.visualizer import start_viser_visualizer

# Trajectory parameters to play with ;)
CIRCLE_RADIUS = 0.5  # [m]
FINGERTIP_HEIGHT = 0.7  # [m]

if __name__ == "__main__":
    robot = pinker.load_robot_description("stretch_description", root_joint="planar")

    # Initialize visualization
    viz = start_viser_visualizer(robot)
    viewer = viz.viewer
    base_frame = viewer.scene.add_frame(
        "/base_frame", axes_length=0.1, axes_radius=0.005
    )
    fingertip_frame = viewer.scene.add_frame(
        "/fingertip_frame", axes_length=0.1, axes_radius=0.005
    )
    base_target_frame = viewer.scene.add_frame(
        "/base_target_frame", axes_length=0.1, axes_radius=0.005
    )
    fingertip_target_frame = viewer.scene.add_frame(
        "/fingertip_target_frame", axes_length=0.1, axes_radius=0.005
    )

    # Define tasks
    base_task = FrameTask(
        "base_link",
        position_cost=0.1,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
    )
    fingertip_task = FrameTask(
        "link_gripper_fingertip_right",
        position_cost=1.0,
        orientation_cost=1e-4,
    )
    tasks = [base_task, fingertip_task]

    # Initialize tasks from the initial configuration
    configuration = pinker.Configuration(robot.model, robot.data, robot.q0)
    base_task.set_target_from_configuration(configuration)
    transform_fingertip_target_to_world = kin.SE3(
        rotation=np.eye(3), translation=np.array([0.0, 0.0, FINGERTIP_HEIGHT])
    ) * configuration.get_transform_frame_to_world(fingertip_task.frame)
    center_translation = transform_fingertip_target_to_world.translation[:2]
    fingertip_task.set_target(transform_fingertip_target_to_world)
    viz.display(configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"

    rate = RateLimiter(frequency=100.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update base target
        T = base_task.transform_target_to_world
        u = np.array([np.cos(t), np.sin(t)])
        T.translation[:2] = center_translation + CIRCLE_RADIUS * u
        T.rotation = kin.rpy_to_matrix(0.0, 0.0, 0.5 * np.pi * t)

        # Update fingertip target
        fingertip_task.transform_target_to_world.translation[2] = (
            FINGERTIP_HEIGHT + 0.1 * u[1]
        )

        # Update visualizer frames
        _T = np.asarray(T.np)
        base_target_frame.position = _T[:3, 3]
        base_target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
        _T = np.asarray(fingertip_task.transform_target_to_world.np)
        fingertip_target_frame.position = _T[:3, 3]
        fingertip_target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
        _T = np.asarray(
            configuration.get_transform_frame_to_world(base_task.frame).np
        )
        base_frame.position = _T[:3, 3]
        base_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
        _T = np.asarray(
            configuration.get_transform_frame_to_world(fingertip_task.frame).np
        )
        fingertip_frame.position = _T[:3, 3]
        fingertip_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz

        # Compute velocity and integrate it into next configuration
        try:
            velocity = solve_ik(configuration, tasks, dt, solver=solver)
        except PinkerError as exn:
            if solver != "quadprog":
                raise PinkerError(
                    "IK failed as detailed in the traceback above. "
                    f"Note that `solve_ik` was called with {solver=}, "
                    "but this example works better with solver='quadprog'."
                ) from exn
            raise exn

        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
