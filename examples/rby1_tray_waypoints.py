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

"""Rainbow Robotics RB-Y1 carrying a tray along planar waypoints."""

import numpy as np
import qpsolvers
import viser.transforms as vtf
from loop_rate_limiters import RateLimiter

import pinker
from pinker import kinematics as kin
from pinker import solve_ik
from pinker.tasks import FrameTask, PostureTask
from pinker.visualizer import start_viser_visualizer

# Trajectory parameters to play with ;)
RADIUS = 1.0  # [m]
SEGMENT_DURATION = 3.0  # [s]

# Figure-eight of quarter circles, as (x, y, heading) waypoints. Each waypoint
# lies on an arc tangent to the heading of the previous one, so that the SE(2)
# geodesic between them has no lateral velocity: a differential-drive base
# like that of the RB-Y1 can follow it.
WAYPOINTS = [
    kin.SE2(kin.exp_so2(theta), [x, y])
    for x, y, theta in [
        (0.0, 0.0, 0.0),
        (RADIUS, RADIUS, 0.5 * np.pi),
        (0.0, 2.0 * RADIUS, np.pi),
        (-RADIUS, RADIUS, -0.5 * np.pi),
        (0.0, 0.0, 0.0),
        (RADIUS, -RADIUS, -0.5 * np.pi),
        (0.0, -2.0 * RADIUS, np.pi),
        (-RADIUS, -RADIUS, 0.5 * np.pi),
    ]
]


def path_at(t: float) -> kin.SE2:
    """Planar pose along the closed path through all waypoints.

    Args:
        t: Time in [s].

    Returns:
        Planar transform from the path frame to the world frame, interpolated
        on the geodesic between the current two waypoints.
    """
    k = int(t // SEGMENT_DURATION) % len(WAYPOINTS)
    alpha = (t % SEGMENT_DURATION) / SEGMENT_DURATION
    start, end = WAYPOINTS[k], WAYPOINTS[(k + 1) % len(WAYPOINTS)]
    return kin.SE2.Interpolate(start, end, alpha)


def lift(M: kin.SE2) -> kin.SE3:
    """Spatial transform of a planar transform in the ground plane.

    Args:
        M: Planar transform.

    Returns:
        Transform rotating about the z-axis and translating in the xy-plane.
    """
    R = np.eye(3)
    R[:2, :2] = M.rotation
    return kin.SE3(R, np.array([*M.translation, 0.0]))


if __name__ == "__main__":
    robot = pinker.load_robot_description(
        "rby1_description", root_joint="planar"
    )

    # Initialize visualization
    viz = start_viser_visualizer(robot)
    viewer = viz.viewer
    path_points = [
        [*path_at(t).translation, 0.01]
        for t in np.arange(0.0, len(WAYPOINTS) * SEGMENT_DURATION, 0.1)
    ]
    viewer.scene.add_spline_catmull_rom(
        "/path", np.array(path_points), closed=True, color=(200, 120, 30)
    )
    tray_box = viewer.scene.add_box(
        "/tray", color=(150, 150, 150), dimensions=(0.3, 0.6, 0.02)
    )

    # Define tasks. The base has no task of its own: inverse kinematics
    # figures out that it needs to drive along the path to keep up with the
    # hands, as the posture task only regularizes the upper body.
    left_hand_task = FrameTask(
        "ee_left",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
    )
    right_hand_task = FrameTask(
        "ee_right",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
    )
    posture_task = PostureTask(
        cost=1e-2,  # [cost] / [rad]
    )
    tasks = [left_hand_task, right_hand_task, posture_task]

    # Start with forearms forward, holding the tray
    q_init = kin.custom_configuration(
        robot.model, left_arm_3=-0.5 * np.pi, right_arm_3=-0.5 * np.pi
    )
    configuration = pinker.Configuration(robot.model, robot.data, q_init)
    posture_task.set_target_from_configuration(configuration)

    # Hand poses in the frame of the path, which moves along the waypoints
    transform_path_to_world = lift(path_at(0.0))
    transform_left_to_path = transform_path_to_world.act_inv(
        configuration.get_transform_frame_to_world("ee_left")
    )
    transform_right_to_path = transform_path_to_world.act_inv(
        configuration.get_transform_frame_to_world("ee_right")
    )

    # The tray rests on the grippers, a few centimeters above the hand frames
    transform_tray_to_path = kin.SE3(
        np.eye(3),
        0.5
        * (
            transform_left_to_path.translation
            + transform_right_to_path.translation
        )
        + np.array([0.0, 0.0, 0.07]),
    )
    transform_left_to_tray = transform_tray_to_path.act_inv(
        transform_left_to_path
    )
    transform_right_to_tray = transform_tray_to_path.act_inv(
        transform_right_to_path
    )
    viz.display(configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=100.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update hand targets
        transform_path_to_world = lift(path_at(t))
        transform_tray_to_world = (
            transform_path_to_world * transform_tray_to_path
        )
        left_hand_task.set_target(
            transform_tray_to_world * transform_left_to_tray
        )
        right_hand_task.set_target(
            transform_tray_to_world * transform_right_to_tray
        )

        # Update visualization of the tray
        tray_box.position = transform_tray_to_world.translation
        tray_box.wxyz = vtf.SO3.from_matrix(
            transform_tray_to_world.rotation
        ).wxyz

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
