# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["daqp", "pinker", "qpsolvers", "robot_descriptions >=3.1.0"]
#
# [tool.uv.sources]
# pinker = { path = "..", editable = true }
# ///

"""Clamp floating-base velocities with FloatingBaseVelocityLimit."""

from __future__ import annotations

import numpy as np
import qpsolvers

import pinker
from pinker.limits import FloatingBaseVelocityLimit
from pinker.tasks import FrameTask


def main() -> None:
    """Run a short IK loop where the base velocity remains bounded."""
    robot = pinker.load_robot_description(
        "upkie_description", root_joint="free_flyer"
    )
    configuration = pinker.Configuration(robot.model, robot.data, robot.q0)

    floating_limit = FloatingBaseVelocityLimit(
        model=robot.model,
        base_frame=None,
        max_linear_velocity=[0.3, 0.3, 0.2],  # [m] / [s]
        max_angular_velocity=[0.8, 0.8, 0.8],  # [rad] / [s]
    )
    configuration.default_limits.append(floating_limit)

    base_task = FrameTask(
        floating_limit.base_frame,
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
    )
    base_task.set_target_from_configuration(configuration)

    # Ask the base to move fast toward a far target. The limit will keep the
    # free-flyer velocity bounded no matter how aggressive the target is.
    transform = base_task.transform_target_to_world.copy()
    transform.translation += np.array([2.0, 0.0, 0.0])
    base_task.set_target(transform)

    dt = 0.1  # [s]
    solver = qpsolvers.available_solvers[0]
    root_joint = robot.model.joints[robot.model.get_joint_id("root_joint")]

    for step in range(10):
        velocity = pinker.solve_ik(
            configuration, [base_task], dt, solver=solver
        )
        base_velocity = velocity[root_joint.idx_v : root_joint.idx_v + 6]
        angular = base_velocity[3:]
        linear = base_velocity[:3]
        print(
            f"step {step:02d} | "
            f"linear = {linear}  [m/s], "
            f"angular = {angular}  [rad/s]"
        )
        configuration.integrate_inplace(velocity, dt)


if __name__ == "__main__":
    main()
