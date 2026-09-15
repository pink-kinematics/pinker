# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["daqp", "pinker", "pycollada", "qpsolvers",
# "robot_descriptions >=3.1.0", "scipy", "trimesh", "viser"]
#
# [tool.uv.sources]
# pinker = { path = "..", editable = true }
# ///

"""Solve IK with the Piper arm end-effector at a prescribed target."""

import sys

import numpy as np
import qpsolvers
import viser.transforms as vtf
from scipy.spatial.transform import Rotation

import pinker
from pinker import kinematics as kin
from pinker.tasks import FrameTask
from pinker.visualizer import start_viser_visualizer

# IK parameters
dt = 1e-2
stop_thres = 1e-8

if __name__ == "__main__":
    robot = pinker.load_robot_description("piper_description")
    model = robot.model

    viz = start_viser_visualizer(robot)
    viewer = viz.viewer
    end_effector_target_frame = viewer.scene.add_frame(
        "/end_effector_target", axes_length=0.1, axes_radius=0.005
    )
    end_effector_frame = viewer.scene.add_frame(
        "/end_effector", axes_length=0.1, axes_radius=0.005
    )

    # Frame details
    joint_name = model.names[-1]
    parent_joint = model.get_joint_id(joint_name)

    FRAME_NAME = "joint6"
    data = kin.Data(model)
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
    kin.forward_kinematics(model, data, q_final)
    target_pose = data.oMi[parent_joint]
    ee_task = FrameTask(FRAME_NAME, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])

    target = kin.SE3.Identity()
    hit_limit = "--hit-limit" in sys.argv
    target.translation = (
        np.array([0.1, 0.2, 0.3]) if hit_limit else np.array([0.0, 0.2, 0.6])
    )
    target.rotation = Rotation.from_euler("xyz", [0, 0, 0]).as_matrix()
    _T = target.toarray()
    end_effector_target_frame.position = _T[:3, 3]
    end_effector_target_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
    ee_task.set_target(target)

    configuration = pinker.Configuration(model, data, q_init)
    viz.display(configuration.q)
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
        q_out = np.clip(q_out, low, high)
        configuration = pinker.Configuration(model, data, q_out)
        kin.update_frame_placements(model, data)
        _T = configuration.get_transform_frame_to_world(
            ee_task.frame
        ).toarray()
        end_effector_frame.position = _T[:3, 3]
        end_effector_frame.wxyz = vtf.SO3.from_matrix(_T[:3, :3]).wxyz
        viz.display(configuration.q)
        error_norm = np.linalg.norm(ee_task.compute_error(configuration))
        nb_steps += 1

    print(f"Terminated after {nb_steps} steps with {error_norm = :.2}")
