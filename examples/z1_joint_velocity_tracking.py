# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["daqp", "loop-rate-limiters", "pinker", "pycollada",
# "qpsolvers", "robot_descriptions >=3.1.0", "trimesh", "viser", "xacrodoc"]
#
# [tool.uv.sources]
# pinker = { path = "..", editable = true }
# ///

"""Unitree Z1 arm tracking a reference joint-velocity trajectory."""

import numpy as np
import qpsolvers
from loop_rate_limiters import RateLimiter

import pinker
from pinker import solve_ik
from pinker.tasks import JointVelocityTask
from pinker.visualizer import start_viser_visualizer

if __name__ == "__main__":
    print(
        "In this example, the arm tracks a sinusoidal joint-velocity "
        "trajectory that is unfeasible at times.\nThe trajectory is only "
        "tracked while the robot stays within joint limits."
    )
    robot = pinker.load_robot_description("z1_description")
    viz = start_viser_visualizer(robot)
    configuration = pinker.Configuration(robot.model, robot.data, robot.q0)
    viz.display(configuration.q)

    # Our only task in this example is a joint-velocity task
    joint_velocity_task = JointVelocityTask(cost=1.0)

    # Select a QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        target_velocity = 2.0 * np.sin(t) * np.ones(robot.nv)
        joint_velocity_task.set_target(target_velocity, dt)
        velocity = solve_ik(
            configuration,
            tasks=[joint_velocity_task],
            dt=dt,
            solver=solver,
        )
        configuration.integrate_inplace(velocity, dt)
        viz.display(configuration.q)
        rate.sleep()
        t += dt
