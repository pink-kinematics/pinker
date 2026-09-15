# SPDX-License-Identifier: Apache-2.0
#
# /// script
# dependencies = ["daqp", "loop-rate-limiters", "meshcat", "pinker",
# "qpsolvers", "robot_descriptions", "xacrodoc"]
# ///

"""Universal Robots UR5 arm tracking a moving target."""

import argparse

import numpy as np
import qpsolvers
import viser.transforms as vtf
from loop_rate_limiters import RateLimiter

import pinker
from pinker import solve_ik
from pinker.barriers import PositionBarrier
from pinker.tasks import FrameTask, PostureTask
from pinker.visualizer import start_viser_visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        "-v",
        help="print out task errors and CBF values during execution",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    robot = pinker.load_robot_description("ur5_official_description")
    viz = start_viser_visualizer(robot)

    end_effector_task = FrameTask(
        "tool0",
        position_cost=10.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
    )

    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )

    pos_barrier = PositionBarrier(
        "tool0",
        indices=[1],
        p_min=np.array([-0.4]),
        p_max=np.array([0.6]),
        gain=np.array([100.0]),
        safe_displacement_gain=1.0,
    )
    barriers = [pos_barrier]

    tasks = [end_effector_task, posture_task]

    q_ref = np.array(
        [
            1.27153374,
            -0.87988708,
            1.89104795,
            1.73996951,
            -0.24610945,
            -0.74979019,
        ]
    )
    configuration = pinker.Configuration(robot.model, robot.data, q_ref)
    for task in tasks:
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    viewer = viz.viewer
    end_effector_target_frame = viewer.scene.add_frame(
        "/end_effector_target", axes_length=0.1, axes_radius=0.005
    )
    end_effector_frame = viewer.scene.add_frame(
        "/end_effector", axes_length=0.1, axes_radius=0.005
    )

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "osqp" in qpsolvers.available_solvers:
        solver = "osqp"

    rate = RateLimiter(frequency=200.0)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update task targets
        end_effector_target = end_effector_task.transform_target_to_world
        end_effector_target.translation[1] = 0.0 + 0.65 * np.sin(t / 4)
        end_effector_target.translation[2] = 0.5

        # Update visualization frames
        _T = np.asarray(end_effector_target.np)
        end_effector_target_frame.position = _T[:3, 3]
        end_effector_target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
        _T = np.asarray(
            configuration.get_transform_frame_to_world(
                end_effector_task.frame
            ).np
        )
        end_effector_frame.position = _T[:3, 3]
        end_effector_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz

        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            barriers=barriers,
        )
        configuration.integrate_inplace(velocity, dt)

        G, h = pos_barrier.compute_qp_inequalities(configuration, dt=dt)
        distance_to_manipulator = configuration.get_transform_frame_to_world(
            "tool0"
        ).translation[1]
        if args.verbose:
            print(
                f"Task error: {end_effector_task.compute_error(configuration)}"
            )
            print(
                "Position CBF value: "
                f"{pos_barrier.compute_barrier(configuration)[0]:0.3f} >= 0"
            )
            print(f"Distance to manipulator: {distance_to_manipulator} <= 0.6")
            print("-" * 60)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
