# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.11"
# dependencies = ["daqp", "loop-rate-limiters", "pinker", "qpsolvers",
# "robot_descriptions", "trimesh", "viser"]
#
# [tool.uv.sources]
# pinker = { path = "..", editable = true }
# robot_descriptions = { git = "https://github.com/robot-descriptions/robot_descriptions.py" }
# ///

"""Microban humanoid with hands, feet and pelvis targets to move around."""

import numpy as np
import qpsolvers
import viser.transforms as vtf
from loop_rate_limiters import RateLimiter

import pinker
from pinker import kinematics as kin
from pinker import solve_ik
from pinker.kinematics import custom_configuration
from pinker.tasks import FrameTask, PostureTask
from pinker.visualizer import start_viser_visualizer

# The model has no hand frames: we add them at the tips of the forearms,
# given by the name of the forearm body and the position of the tip in its
# frame (forearms point down in the neutral configuration).
HAND_FRAMES = {
    "left_hand": ("radius_2", [0.0, 0.07, -0.0135]),  # [m]
    "right_hand": ("radius", [0.0, -0.07, 0.0135]),  # [m]
}


def add_hand_frames(model: kin.Model) -> None:
    """Add operational frames at the tips of the forearms of Microban.

    Args:
        model: Robot model to add the frames to.
    """
    for hand, (forearm, position) in HAND_FRAMES.items():
        forearm_id = model.get_frame_id(forearm)
        forearm_frame = model.frames[forearm_id]
        model.add_frame(
            kin.Frame(
                hand,
                forearm_frame.parent_joint,
                forearm_id,
                forearm_frame.placement
                * kin.SE3(np.eye(3), np.array(position)),
                kin.FrameType.OP_FRAME,
            )
        )


if __name__ == "__main__":
    robot = pinker.load_robot_description("microban_mj_description")
    add_hand_frames(robot.model)
    viz = start_viser_visualizer(robot)  # after adding frames
    viewer = viz.viewer

    # Start with bent knees, away from the singularity of straight legs, and
    # lift the floating base so that the soles are on the ground
    q_init = custom_configuration(
        robot.model,
        left_hip_pitch=-0.3,
        left_knee=0.6,
        left_ankle_pitch=-0.3,
        right_hip_pitch=-0.3,
        right_knee=0.6,
        right_ankle_pitch=-0.3,
    )
    data = robot.model.create_data()  # with the hand frames
    configuration = pinker.Configuration(robot.model, data, q_init)
    q_init[2] -= configuration.get_transform_frame_to_world(
        "left_foot"
    ).translation[2]
    configuration = pinker.Configuration(robot.model, data, q_init)
    viz.display(configuration.q)

    pelvis_task = FrameTask(
        "trunk",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=1.0,
    )
    foot_tasks = [
        FrameTask(
            foot,
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
            lm_damping=1.0,
        )
        for foot in ("left_foot", "right_foot")
    ]
    hand_tasks = [
        FrameTask(
            hand,
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=0.0,  # [cost] / [rad]
            lm_damping=1.0,
        )
        for hand in HAND_FRAMES
    ]
    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )
    frame_tasks = [pelvis_task, *foot_tasks, *hand_tasks]
    for task in frame_tasks:
        task.set_target_from_configuration(configuration)
    posture_task.set_target_from_configuration(configuration)
    tasks = [*frame_tasks, posture_task]

    # Interactive handles to move frame targets around in the scene. Hand
    # handles only translate as hand tasks have no orientation cost.
    target_handles = []
    for task in frame_tasks:
        target = task.transform_target_to_world
        handle = viewer.scene.add_transform_controls(
            f"/targets/{task.frame}",
            position=target.translation,
            wxyz=vtf.SO3.from_matrix(target.rotation).wxyz,
            scale=0.05,
            disable_rotations=task in hand_tasks,
        )
        target_handles.append((task, handle))
    viewer.scene.add_grid("/grid", width=0.5, height=0.5, cell_size=0.05)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    while True:
        for task, handle in target_handles:
            task.set_target(
                kin.SE3(
                    vtf.SO3(np.array(handle.wxyz)).as_matrix(),
                    np.array(handle.position),
                )
            )
        velocity = solve_ik(
            configuration, tasks, dt, solver=solver, safety_break=False
        )
        configuration.integrate_inplace(velocity, dt)
        viz.display(configuration.q)
        rate.sleep()
