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

"""Upkie wheeled biped rolling around."""

import meshcat_shapes
import numpy as np
import pinocchio as pin
import qpsolvers
import viser.transforms as vtf
from loop_rate_limiters import RateLimiter

import pinker
from pinker import solve_ik
from pinker.tasks import FrameTask, RollingTask
from pinker.visualizer import start_viser_visualizer

if __name__ == "__main__":
    robot = pinker.load_robot_description("upkie_description", root_joint="free_flyer")
    visualizer = start_viser_visualizer(robot)

    base_task = FrameTask(
        "base",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
    )
    left_wheel_rolling = RollingTask(
        "left_wheel",
        floor_frame="universe",
        wheel_radius=0.06,
        cost=10.0,
    )
    right_wheel_rolling = RollingTask(
        "right_wheel",
        floor_frame="universe",
        wheel_radius=0.06,
        cost=10.0,
    )
    left_wheel_position = FrameTask(
        "left_wheel",
        position_cost=1.0,
        orientation_cost=0.0,
    )
    right_wheel_position = FrameTask(
        "right_wheel",
        position_cost=1.0,
        orientation_cost=0.0,
    )

    init_x = -2.0  # in [m]
    q_init = robot.q0.copy()
    q_init[0] = init_x  # world x-axis
    q_init[2] = 0.56  # world z-axis
    configuration = pinker.Configuration(robot.model, robot.data, q_init)
    base_task.set_target_from_configuration(configuration)
    left_wheel_position.set_target_from_configuration(configuration)
    right_wheel_position.set_target_from_configuration(configuration)
    visualizer.display(configuration.q)

    base_target = base_task.transform_target_to_world
    left_wheel_target = left_wheel_position.transform_target_to_world
    right_wheel_target = right_wheel_position.transform_target_to_world

    viewer = visualizer.viewer
    base_frame = viewer.scene.add_frame(
        "/base", axes_length=0.1, axes_radius=0.005
    )
    base_target_frame = viewer.scene.add_frame(
        "/base_target", axes_length=0.1, axes_radius=0.005
    )
    left_wheel_target_frame = viewer.scene.add_frame(
        "/left_wheel_target", axes_length=0.1, axes_radius=0.005
    )
    right_wheel_target_frame = viewer.scene.add_frame(
        "/right_wheel_target", axes_length=0.1, axes_radius=0.005
    )

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "proxqp" in qpsolvers.available_solvers:
        solver = "proxqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]

    while True:
        # Update base task target
        base_x = 0.1 * t + init_x
        base_target.translation[0] = base_x
        left_wheel_target.translation[0] = base_x + 0.1 * np.sin(t)
        right_wheel_target.translation[0] = base_x - 0.1 * np.sin(t)

        # Update visualization frames
        _T = np.asarray(base_target.np)
        base_target_frame.position = _T[:3, 3]
        base_target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
        _T = np.asarray(left_wheel_target.np)
        left_wheel_target_frame.position = _T[:3, 3]
        left_wheel_target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
        _T = np.asarray(right_wheel_target.np)
        right_wheel_target_frame.position = _T[:3, 3]
        right_wheel_target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
        _T = np.asarray(configuration.get_transform_frame_to_world("base").np)
        base_frame.position = _T[:3, 3]
        base_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(
            configuration,
            tasks=[base_task, left_wheel_position, right_wheel_position],
            dt=dt,
            solver=solver,
            damping=1e-3,
            constraints=[left_wheel_rolling, right_wheel_rolling],
        )
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        visualizer.display(configuration.q)
        rate.sleep()
        t += dt
