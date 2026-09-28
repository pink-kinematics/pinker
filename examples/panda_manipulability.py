# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["daqp", "loop-rate-limiters", "matplotlib", "pinker",
# "pycollada", "qpsolvers", "robot_descriptions >=3.1.0", "trimesh", "viser"]
#
# [tool.uv.sources]
# pinker = { path = "..", editable = true }
# ///

"""Two Pandas tracking the same target, one increasing manipulability."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import qpsolvers
import viser
import viser.transforms as vtf
from loop_rate_limiters import RateLimiter

import pinker
from pinker import solve_ik
from pinker.kinematics import custom_configuration
from pinker.tasks import FrameTask, ManipulabilityTask, PostureTask
from pinker.visualizer import ViserVisualizer, start_viser_visualizer

NB_STEPS = 2000  # number of steps to run the example for
SWEEP_PERIOD = 20.0  # [s] duration of one target sweep around the base
SWEEP_RADIUS = 0.45  # [m] distance from the base to the target
BASELINE_OFFSET = 1.0  # [m] lateral offset of the baseline robot in the scene


def target_pose(
    t: float, rotation_home: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep the target around the base of the robot.

    Args:
        t: Time in seconds.
        rotation_home: Orientation of the end effector at the home
            configuration, which the target keeps while it sweeps around the
            vertical axis.

    Returns:
        Target pose at that time, as a (rotation, translation) pair.
    """
    angle = 2.0 * np.pi * t / SWEEP_PERIOD
    cos, sin = np.cos(angle), np.sin(angle)
    rotation_z = np.array(
        [
            [cos, -sin, 0.0],
            [sin, cos, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array(
        [
            SWEEP_RADIUS * cos,  # [m]
            SWEEP_RADIUS * sin,  # [m]
            0.45 + 0.15 * np.sin(2.0 * angle),  # [m]
        ]
    )
    return rotation_z @ rotation_home, translation


if __name__ == "__main__":
    robot = pinker.load_robot_description("panda_description")
    frame_name = "panda_hand_tcp"

    # Both robots are displayed in the same scene, the baseline one being
    # shifted sideways by moving the root node of its visuals
    viz = start_viser_visualizer(robot)
    viewer = viz.viewer
    baseline_viz = ViserVisualizer(robot.model, robot.visual_model)
    baseline_viz.init_viewer(viewer=viewer)
    baseline_viz.load_viewer_model(rootNodeName="baseline_robot")
    baseline_viz.visual_root_frame.position = (0.0, BASELINE_OFFSET, 0.0)
    viewer.scene.add_label(
        "/manipulability_label",
        "Manipulability task",
        position=(0.0, 0.0, 1.0),
    )
    viewer.scene.add_label(
        "/baseline_label",
        "Baseline",
        position=(0.0, BASELINE_OFFSET, 1.0),
    )

    # Same tasks for both robots, except for the manipulability one
    end_effector_task = FrameTask(
        frame_name,
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=1.0,
    )
    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )
    manipulability_task = ManipulabilityTask(
        frame_name,
        robot.model,
        cost=0.3,  # [cost] * [s] / [manipulability]
        lm_damping=1e-3,
        manipulability_rate=1.5,  # [manipulability] / [s]
    )
    tasks = [end_effector_task, posture_task, manipulability_task]

    baseline_end_effector_task = FrameTask(
        frame_name,
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=1.0,
    )
    baseline_posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )
    baseline_tasks = [baseline_end_effector_task, baseline_posture_task]

    # Both robots start from the same configuration, each with its own data
    q_ref = custom_configuration(
        robot.model,
        panda_joint1=0.0,
        panda_joint2=-0.785398,
        panda_joint3=0.0,
        panda_joint4=-2.35619,
        panda_joint5=0.0,
        panda_joint6=1.5708,
        panda_joint7=0.785398,
    )
    configuration = pinker.Configuration(robot.model, robot.data, q_ref)
    baseline_configuration = configuration.copy()
    for task in (end_effector_task, posture_task):
        task.set_target_from_configuration(configuration)
    for task in (baseline_end_effector_task, baseline_posture_task):
        task.set_target_from_configuration(baseline_configuration)
    rotation_home = configuration.get_transform_frame_to_world(
        frame_name
    ).rotation

    # Target frames are children of the visuals of each robot, so that the one
    # of the baseline robot is shifted along with it
    target_frame = viewer.scene.add_frame(
        viz.visual_root_node_name + "/target",
        axes_length=0.1,
        axes_radius=0.005,
    )
    baseline_target_frame = viewer.scene.add_frame(
        baseline_viz.visual_root_node_name + "/target",
        axes_length=0.1,
        axes_radius=0.005,
    )

    cost_slider = viewer.gui.add_slider(
        "Manipulability cost",
        min=0.0,
        max=1.0,
        initial_value=float(manipulability_task.cost),
        step=0.05,
    )
    rate_slider = viewer.gui.add_slider(
        "Manipulability rate",
        min=-5.0,
        max=20.0,
        initial_value=manipulability_task.manipulability_rate,
        step=0.5,
    )

    @cost_slider.on_update
    def _(event: viser.GuiEvent[viser.GuiSliderHandle]) -> None:
        manipulability_task.cost = event.target.value

    @rate_slider.on_update
    def _(event: viser.GuiEvent[viser.GuiSliderHandle]) -> None:
        manipulability_task.manipulability_rate = event.target.value

    viz.display(configuration.q)
    baseline_viz.display(baseline_configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=100.0, warn=False)
    dt = rate.period
    manipulabilities, baseline_manipulabilities = [], []
    errors, baseline_errors = [], []
    times = []
    t = 0.0  # [s]
    for step in range(NB_STEPS):
        # Update task targets
        rotation, translation = target_pose(t, rotation_home)
        for task in (end_effector_task, baseline_end_effector_task):
            task.transform_target_to_world.rotation = rotation
            task.transform_target_to_world.translation = translation

        # Update visualization frames
        for frame in (target_frame, baseline_target_frame):
            frame.position = translation
            frame.wxyz = vtf.SO3.from_matrix(rotation).wxyz

        # Compute velocities and integrate them into next configurations
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)
        baseline_velocity = solve_ik(
            baseline_configuration, baseline_tasks, dt, solver=solver
        )
        baseline_configuration.integrate_inplace(baseline_velocity, dt)

        # Append plotting data to lists
        manipulabilities.append(
            manipulability_task.compute_manipulability(configuration)
        )
        baseline_manipulabilities.append(
            manipulability_task.compute_manipulability(baseline_configuration)
        )
        errors.append(
            np.linalg.norm(end_effector_task.compute_error(configuration))
        )
        baseline_errors.append(
            np.linalg.norm(
                baseline_end_effector_task.compute_error(
                    baseline_configuration
                )
            )
        )
        times.append(t)

        # Visualize results at fixed FPS
        viz.display(configuration.q)
        baseline_viz.display(baseline_configuration.q)
        rate.sleep()
        t += dt

    # Plot manipulability and tracking error over time
    _, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(times, manipulabilities, label="Manipulability task")
    axes[0].plot(times, baseline_manipulabilities, "--", label="Baseline")
    axes[0].set_title("Yoshikawa manipulability index")
    axes[0].set_ylabel("Manipulability")
    axes[0].legend()
    axes[1].plot(times, errors, label="Manipulability task")
    axes[1].plot(times, baseline_errors, "--", label="Baseline")
    axes[1].set_title("End-effector task error")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Error norm")
    axes[1].legend()
    plt.show()
