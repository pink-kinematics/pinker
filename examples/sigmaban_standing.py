# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["daqp", "loop-rate-limiters", "pinker", "pycollada",
# "qpsolvers", "robot_descriptions >=3.1.0", "trimesh", "viser"]
#
# [tool.uv.sources]
# pinker = { path = "..", editable = true }
# ///

"""SigmaBan humanoid standing on two feet."""

import numpy as np
import qpsolvers
import viser.transforms as vtf
from loop_rate_limiters import RateLimiter

import pinker
from pinker import kinematics as kin
from pinker import solve_ik
from pinker.tasks import FrameTask, PostureTask
from pinker.visualizer import start_viser_visualizer

if __name__ == "__main__":
    robot = pinker.load_robot_description(
        "sigmaban_description", root_joint="free_flyer"
    )

    # Initialize visualization
    viz = start_viser_visualizer(robot)
    viewer = viz.viewer
    left_foot_target_frame = viewer.scene.add_frame(
        "/left_foot_target", axes_length=0.1, axes_radius=0.005
    )
    right_foot_target_frame = viewer.scene.add_frame(
        "/right_foot_target", axes_length=0.1, axes_radius=0.005
    )
    torso_target_frame = viewer.scene.add_frame(
        "/torso_target", axes_length=0.1, axes_radius=0.005
    )

    configuration = pinker.Configuration(robot.model, robot.data, robot.q0)
    viz.display(configuration.q)

    left_foot_task = FrameTask(
        "left_foot_tip",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    torso_task = FrameTask(
        "torso",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    right_foot_task = FrameTask(
        "right_foot_tip",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    posture_task = PostureTask(
        cost=1e-2,  # [cost] / [rad]
    )
    tasks = [left_foot_task, torso_task, right_foot_task, posture_task]

    torso_pose = configuration.get_transform_frame_to_world("torso").copy()
    # torso_pose.translation[0] += 0.05
    torso_task.set_target(torso_pose)
    posture_task.set_target_from_configuration(configuration)

    transform_left_foot_tip_target_to_init = kin.SE3(
        np.eye(3), np.array([0.0, 0.03, 0.0])
    )
    transform_right_foot_tip_target_to_init = kin.SE3(
        np.eye(3), np.array([0.0, -0.03, 0.0])
    )

    left_foot_task.set_target(
        configuration.get_transform_frame_to_world("left_foot_tip")
        * transform_left_foot_tip_target_to_init
    )
    right_foot_task.set_target(
        configuration.get_transform_frame_to_world("right_foot_tip")
        * transform_right_foot_tip_target_to_init
    )
    torso_task.set_target(configuration.get_transform_frame_to_world("torso"))

    # Display targets
    _T = left_foot_task.transform_target_to_world.toarray()
    left_foot_target_frame.position = _T[:3, 3]
    left_foot_target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
    _T = right_foot_task.transform_target_to_world.toarray()
    right_foot_target_frame.position = _T[:3, 3]
    right_foot_target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
    _T = torso_task.transform_target_to_world.toarray()
    torso_target_frame.position = _T[:3, 3]
    torso_target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
