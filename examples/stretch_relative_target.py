# SPDX-License-Identifier: Apache-2.0
#
# /// script
# dependencies = ["clarabel", "loop-rate-limiters", "meshcat", "pinker",
# "qpsolvers", "robot_descriptions"]
# ///

"""Move a Stretch RE1 with a fingertip target in the mobile-base frame."""

import numpy as np
import qpsolvers
import viser.transforms as vtf
from loop_rate_limiters import RateLimiter

import pinker
from pinker import PinkerError, solve_ik
from pinker import kinematics as kin
from pinker.tasks import FrameTask, RelativeFrameTask
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
    fingertip_task = RelativeFrameTask(
        "link_gripper_fingertip_right",
        "base_link",
        position_cost=1.0,
        orientation_cost=1e-4,
    )
    tasks = [base_task, fingertip_task]

    # Initialize tasks from the initial configuration
    configuration = pinker.Configuration(robot.model, robot.data, robot.q0)
    base_task.set_target_from_configuration(configuration)
    transform_fingertip_target_to_base = kin.SE3(
        rotation=np.eye(3),
        translation=np.array([0.1, 0.0, FINGERTIP_HEIGHT]),
    )
    transform_fingertip_to_world = configuration.get_transform_frame_to_world(
        fingertip_task.frame
    )
    center_translation = transform_fingertip_to_world.translation[:2]
    fingertip_task.set_target(transform_fingertip_target_to_base)
    viz.display(configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"

    rate = RateLimiter(frequency=100.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update base task target
        T = base_task.transform_target_to_world
        u = np.array([np.cos(t), np.sin(t)])
        T.translation[:2] = center_translation + CIRCLE_RADIUS * u
        T.rotation = kin.rpy_to_matrix(0.0, 0.0, 0.5 * np.pi * t)

        # Update fingertip target
        fingertip_task.transform_target_to_root.translation[2] = (
            FINGERTIP_HEIGHT + 0.1 * u[1]
        )

        # Update visualizer frames
        _T = np.asarray(T.np)
        base_target_frame.position = _T[:3, 3]
        base_target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
        transform_fingertip_target_to_world = (
            configuration.get_transform_frame_to_world("base_link")
            * fingertip_task.transform_target_to_root
        )
        _T = np.asarray(transform_fingertip_target_to_world.np)
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
        # input("next?")
        t += dt
