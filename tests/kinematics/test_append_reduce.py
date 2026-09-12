#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for appendModel, buildReducedModel and the visuals parser."""

import os
import tempfile
import unittest

import numpy as np
import pinocchio as pin

from pinker import kinematics as mq

from compare import (
    assert_kinematics_equal,
    assert_liegroup_equal,
    assert_models_equal,
    random_configuration,
)

HERE = os.path.dirname(__file__)
WHEELED_URDF = os.path.join(HERE, "wheeled.urdf")

LOCKED_JOINTS = [
    ["slider_joint"],
    ["left_wheel_joint", "right_wheel_joint"],
    ["head_joint", "left_wheel_joint"],
]

VISUAL_URDF = """<?xml version="1.0"?>
<robot name="visual_test">
  <material name="shared"><color rgba="0.1 0.2 0.3 1.0"/></material>
  <link name="base">
    <visual>
      <origin xyz="0.1 0 0.2" rpy="0 0.5 0"/>
      <geometry><box size="0.3 0.2 0.1"/></geometry>
      <material name="shared"/>
    </visual>
    <visual>
      <geometry><sphere radius="0.05"/></geometry>
      <material name="inline"><color rgba="1 0 0 0.5"/></material>
    </visual>
  </link>
  <link name="rod">
    <visual>
      <geometry><cylinder radius="0.02" length="0.5"/></geometry>
    </visual>
  </link>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="rod"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def build_arm(backend):
    """Two-joint arm model built with either backend."""
    model = backend.Model()
    placement_1 = backend.SE3(np.eye(3), np.array([0.0, 0.0, 0.1]))
    j1 = model.addJoint(0, backend.JointModelRY(), placement_1, "shoulder")
    model.addJointFrame(j1)
    inertia = backend.Inertia(1.5, np.array([0.0, 0.0, 0.2]), np.eye(3))
    model.appendBodyToJoint(j1, inertia, backend.SE3.Identity())
    model.addBodyFrame("upper_arm", j1, backend.SE3.Identity(), -1)
    placement_2 = backend.SE3(np.eye(3), np.array([0.0, 0.0, 0.4]))
    j2 = model.addJoint(j1, backend.JointModelRX(), placement_2, "elbow")
    model.addJointFrame(j2)
    model.appendBodyToJoint(j2, inertia, backend.SE3.Identity())
    model.addBodyFrame("forearm", j2, backend.SE3.Identity(), -1)
    return model


class TestAppendReduce(unittest.TestCase):
    """Test model edition against Pinocchio's."""

    def test_reduced_model_matches_pinocchio(self):
        """Reduced models match Pinocchio's buildReducedModel."""
        for lock_names in LOCKED_JOINTS:
            with self.subTest(locked=",".join(lock_names)):
                mp = pin.buildModelFromUrdf(
                    WHEELED_URDF, pin.JointModelFreeFlyer()
                )
                mm = mq.buildModelFromUrdf(
                    WHEELED_URDF, mq.JointModelFreeFlyer()
                )
                rng = np.random.default_rng(3)
                q_ref = random_configuration(mp, rng)
                ids_p = [mp.getJointId(name) for name in lock_names]
                ids_m = [mm.getJointId(name) for name in lock_names]
                reduced_p = pin.buildReducedModel(mp, ids_p, q_ref)
                reduced_m = mq.buildReducedModel(mm, ids_m, q_ref)
                assert_models_equal(reduced_p, reduced_m)
                for _ in range(5):
                    q = random_configuration(reduced_p, rng)
                    assert_kinematics_equal(reduced_p, reduced_m, q)
                    assert_liegroup_equal(reduced_p, reduced_m, q, rng)

    def test_append_model_matches_pinocchio(self):
        """Appending an arm to a floating base matches Pinocchio."""
        arm_p, arm_m = build_arm(pin), build_arm(mq)
        base_p = pin.buildModelFromUrdf(
            WHEELED_URDF, pin.JointModelFreeFlyer()
        )
        base_m = mq.buildModelFromUrdf(WHEELED_URDF, mq.JointModelFreeFlyer())
        placement = pin.SE3(np.eye(3), np.array([0.1, 0.0, 0.2]))
        merged_p = pin.appendModel(
            base_p, arm_p, base_p.getFrameId("head"), placement
        )
        merged_m = mq.appendModel(
            base_m,
            arm_m,
            base_m.getFrameId("head"),
            mq.SE3(placement.rotation, placement.translation),
        )
        assert_models_equal(merged_p, merged_m)
        rng = np.random.default_rng(4)
        for _ in range(5):
            q = random_configuration(merged_p, rng)
            assert_kinematics_equal(merged_p, merged_m, q)
            assert_liegroup_equal(merged_p, merged_m, q, rng)

    def test_visuals_parser(self):
        """URDF visuals are parsed with shapes, colors and placements."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            urdf_path = os.path.join(tmp_dir, "visual_test.urdf")
            with open(urdf_path, "w", encoding="utf-8") as urdf_file:
                urdf_file.write(VISUAL_URDF)
            model = mq.buildModelFromUrdf(urdf_path)
            geometry = mq.buildGeomFromUrdf(model, urdf_path)
        self.assertEqual(geometry.ngeoms, 3)
        box, sphere, cylinder = geometry.geometryObjects
        self.assertEqual(box.shape, "box")
        self.assertTrue(np.allclose(box.size, [0.3, 0.2, 0.1]))
        self.assertTrue(np.allclose(box.meshColor, [0.1, 0.2, 0.3, 1.0]))
        self.assertTrue(
            np.allclose(box.placement.translation, [0.1, 0.0, 0.2])
        )
        self.assertEqual(sphere.shape, "sphere")
        self.assertTrue(np.allclose(sphere.meshColor, [1.0, 0.0, 0.0, 0.5]))
        self.assertEqual(cylinder.shape, "cylinder")
        self.assertEqual(cylinder.parentJoint, model.getJointId("hinge"))

    def test_reduced_robot_wrapper(self):
        """RobotWrapper.buildReducedRobot locks joints by name."""
        robot = mq.RobotWrapper.BuildFromURDF(WHEELED_URDF)
        reduced = robot.buildReducedRobot(
            ["left_wheel_joint", "slider_joint"]
        )
        # continuous joint has nq=2
        self.assertEqual(reduced.model.nq, robot.model.nq - 3)
        self.assertEqual(reduced.model.nv, robot.model.nv - 2)
        self.assertFalse(reduced.model.existJointName("slider_joint"))
        # the locked joint is now a fixed frame
        self.assertTrue(reduced.model.existFrame("slider_joint"))
