# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["clarabel", "loop-rate-limiters", "pinker", "pycollada",
# "qpsolvers", "robot_descriptions >=3.1.0", "trimesh", "typing-extensions",
# "viser"]
#
# [tool.uv.sources]
# pinker = { path = "..", editable = true }
# ///

"""JVRC-1 humanoid standing on two feet and reaching with a hand."""

import numpy as np
import qpsolvers
import viser.transforms as vtf
from loop_rate_limiters import RateLimiter

import pinker
from pinker import kinematics as kin
from pinker import solve_ik
from pinker.tasks import FrameTask
from pinker.visualizer import start_viser_visualizer


class WavingPose:
    """Moving target to the wave the right hand."""

    def __init__(self, init: kin.SE3):
        """Initialize pose.

        Args:
            init: Initial transform from the wrist frame to the world frame.
        """
        self.init = init

    def at(self, t):
        """Get waving pose at a given time.

        Args:
            t: Time in seconds.
        """
        T = self.init.copy()
        R = T.rotation
        R = np.dot(R, kin.rpy_to_matrix(0.0, 0.0, np.pi / 2))
        R = np.dot(R, kin.rpy_to_matrix(0.0, -np.pi, 0.0))
        T.rotation = R
        T.translation[0] += 0.5
        T.translation[1] += -0.1 + 0.05 * np.sin(8.0 * t)
        T.translation[2] += 0.5
        return T


if __name__ == "__main__":
    robot = pinker.load_robot_description("jvrc_description", root_joint="free_flyer")

    # Initialize visualization
    viz = start_viser_visualizer(robot)
    viewer = viz.viewer
    wrist_frame = viewer.scene.add_frame(
        "/right_wrist_pose", axes_length=0.1, axes_radius=0.005
    )
    pelvis_pose_frame = viewer.scene.add_frame(
        "/pelvis_pose", axes_length=0.1, axes_radius=0.005
    )

    configuration = pinker.Configuration(robot.model, robot.data, robot.q0)
    viz.display(configuration.q)

    left_foot_task = FrameTask(
        "l_ankle", position_cost=1.0, orientation_cost=3.0
    )
    pelvis_task = FrameTask(
        "PELVIS_S", position_cost=1.0, orientation_cost=0.0
    )
    right_foot_task = FrameTask(
        "r_ankle", position_cost=1.0, orientation_cost=3.0
    )
    right_wrist_task = FrameTask(
        "r_wrist", position_cost=1.0, orientation_cost=3.0
    )
    tasks = [left_foot_task, pelvis_task, right_foot_task, right_wrist_task]

    pelvis_pose = configuration.get_transform_frame_to_world("PELVIS_S").copy()
    pelvis_pose.translation[0] += 0.05
    _T = np.asarray(pelvis_pose.np)
    pelvis_pose_frame.position = _T[:3, 3]
    pelvis_pose_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
    pelvis_task.set_target(pelvis_pose)

    transform_l_ankle_target_to_init = kin.SE3(
        np.eye(3), np.array([0.1, 0.0, 0.0])
    )
    transform_r_ankle_target_to_init = kin.SE3(
        np.eye(3), np.array([-0.1, 0.0, 0.0])
    )

    left_foot_task.set_target(
        configuration.get_transform_frame_to_world("l_ankle")
        * transform_l_ankle_target_to_init
    )
    right_foot_task.set_target(
        configuration.get_transform_frame_to_world("r_ankle")
        * transform_r_ankle_target_to_init
    )
    pelvis_task.set_target(
        configuration.get_transform_frame_to_world("PELVIS_S")
    )

    right_wrist_pose = WavingPose(
        configuration.get_transform_frame_to_world("r_wrist")
    )

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "proxqp" in qpsolvers.available_solvers:
        solver = "proxqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update task targets
        right_wrist_task.set_target(right_wrist_pose.at(t))
        _T = np.asarray(right_wrist_pose.at(t).np)
        wrist_frame.position = _T[:3, 3]
        wrist_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
