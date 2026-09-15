# SPDX-License-Identifier: Apache-2.0
#
# /// script
# dependencies = [
#     "daqp",
#     "pinker",
#     "qpsolvers",
#     "robot_descriptions",
#     "xacrodoc",
# ]
# ///

"""Solve IK with the UR10 arm end-effector at a prescribed target."""

import numpy as np
import qpsolvers

import pinker
from pinker import kinematics as kin
from pinker.tasks import FrameTask

# IK parameters
dt = 1e-2
stop_thres = 1e-8

if __name__ == "__main__":
    robot = pinker.load_robot_description("ur10_official_description")
    model = robot.model

    # Frame details
    joint_name = model.names[-1]
    parent_joint = model.get_joint_id(joint_name)
    parent_frame = model.get_frame_id(joint_name)
    placement = kin.SE3.Identity()

    FRAME_NAME = "ee_frame"
    ee_frame = model.add_frame(
        kin.Frame(
            FRAME_NAME,
            parent_joint,
            parent_frame,
            placement,
            kin.FrameType.OP_FRAME,
        )
    )
    robot.data = kin.Data(model)
    low = model.lower_position_limit
    high = model.upper_position_limit
    q_init = kin.neutral(model)

    # Task details
    np.random.seed(0)
    q_final = np.array(
        [
            np.random.uniform(low=low[i], high=high[i], size=(1,))[0]
            for i in range(model.nq)
        ]
    )
    kin.forward_kinematics(model, robot.data, q_final)
    target_pose = robot.data.oMi[parent_joint]
    ee_task = FrameTask(FRAME_NAME, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    ee_task.set_target(target_pose)

    configuration = pinker.Configuration(model, robot.data, q_init)
    error_norm = np.linalg.norm(ee_task.compute_error(configuration))
    print(f"Starting from {error_norm = :.2}")
    print(f"Desired precision is error_norm < {stop_thres}")

    nb_steps = 0
    while error_norm > stop_thres:
        dv = pinker.solve_ik(
            configuration,
            tasks=[ee_task],
            dt=dt,
            damping=1e-8,
            solver=(
                "daqp"
                if "daqp" in qpsolvers.available_solvers
                else qpsolvers.available_solvers[0]
            ),
        )
        q_out = kin.integrate(model, configuration.q, dv * dt)
        configuration = pinker.Configuration(model, robot.data, q_out)
        kin.update_frame_placements(model, robot.data)
        error_norm = np.linalg.norm(ee_task.compute_error(configuration))
        nb_steps += 1

    print(f"Terminated after {nb_steps} steps with {error_norm = :.2}")
