# SPDX-License-Identifier: Apache-2.0
#
# /// script
# dependencies = ["daqp", "loop-rate-limiters", "meshcat", "pinker",
# "qpsolvers", "robot_descriptions"]
# ///

"""Upkie wheeled biped bending its knees."""

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
    robot = pinker.load_robot_description("upkie_description")

    # A large posture cost on the wheels keeps them locked in place, as this
    # example only bends the knees. Other joints are only regularized.
    posture_cost = np.full(robot.model.nv, 1e-3)  # [cost] / [rad]
    for wheel in ("left_wheel", "right_wheel"):
        joint = robot.model.joints[robot.model.get_joint_id(wheel)]
        posture_cost[joint.idx_v] = 1.0

    # Initialize visualization
    viz = start_viser_visualizer(robot)
    viewer = viz.viewer
    left_contact_target_frame = viewer.scene.add_frame(
        "/left_contact_target", axes_length=0.1, axes_radius=0.005
    )
    right_contact_target_frame = viewer.scene.add_frame(
        "/right_contact_target", axes_length=0.1, axes_radius=0.005
    )
    left_contact_frame = viewer.scene.add_frame(
        "/left_contact", axes_length=0.1, axes_radius=0.005
    )
    right_contact_frame = viewer.scene.add_frame(
        "/right_contact", axes_length=0.1, axes_radius=0.005
    )

    tasks = {
        "left_contact": FrameTask(
            "left_contact",
            position_cost=[0.1, 0.0, 0.1],  # [cost] / [m]
            orientation_cost=0.0,  # [cost] / [rad]
        ),
        "right_contact": FrameTask(
            "right_contact",
            position_cost=[0.1, 0.0, 0.1],  # [cost] / [m]
            orientation_cost=0.0,  # [cost] / [rad]
        ),
        "posture": PostureTask(
            cost=posture_cost,  # [cost] / [rad]
        ),
    }

    q_ref = custom_configuration(
        robot.model,
        left_hip=-0.2,
        left_knee=0.4,
        right_hip=0.2,
        right_knee=-0.4,
    )
    configuration = pinker.Configuration(robot.model, robot.data, q_ref)
    for body, task in tasks.items():
        if type(task) is FrameTask:
            task.set_target_from_configuration(configuration)
    tasks["posture"].set_target(q_ref)
    viz.display(configuration.q)

    left_contact_target = tasks["left_contact"].transform_target_to_world
    right_contact_target = tasks["right_contact"].transform_target_to_world

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update task targets
        left_contact_target.translation[2] += 0.1 * np.sin(t) * dt
        right_contact_target.translation[2] += 0.1 * np.sin(t) * dt

        # Update visualization frames
        _T = np.asarray(left_contact_target.np)
        left_contact_target_frame.position = _T[:3, 3]
        left_contact_target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
        _T = np.asarray(right_contact_target.np)
        right_contact_target_frame.position = _T[:3, 3]
        right_contact_target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
        for body_frame, body in (
            (left_contact_frame, "left_contact"),
            (right_contact_frame, "right_contact"),
        ):
            _T = np.asarray(
                configuration.get_transform_frame_to_world(body).np
            )
            body_frame.position = _T[:3, 3]
            body_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks.values(), dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
