# SPDX-License-Identifier: Apache-2.0

"""Test fixture for other library features."""

import unittest

import numpy as np

from pinker import kinematics as kin
from pinker.exceptions import ConfigurationError
from pinker.utils import VectorSpace, custom_configuration_vector

from .loaders import load_robot_description


class TestUtils(unittest.TestCase):
    """Test utility classes and functions."""

    def test_custom_configuration_vector(self):
        """Check a custom configuration vector for Upkie.

        Assumes the left and right knees have joint indices respectively 8 and
        11 in the configuration vector.
        """
        robot = load_robot_description(
            "upkie_description", root_joint=kin.JointModelFreeFlyer()
        )
        q = custom_configuration_vector(robot, left_knee=0.2, right_knee=-0.2)
        self.assertAlmostEqual(q[8], 0.2)
        self.assertAlmostEqual(q[11], -0.2)

    def test_custom_configuration_vector_unbounded_joints(self):
        """Single number for an unbounded joint should yield an error."""
        robot = load_robot_description("gen2_description", root_joint=None)
        with self.assertRaises(ConfigurationError):
            custom_configuration_vector(robot, j2s6s200_joint_1=0.0)
        cos = np.sqrt(3) / 2.0
        sin = 0.5
        q = custom_configuration_vector(robot, j2s6s200_joint_1=[cos, sin])
        self.assertAlmostEqual(q[0], cos)
        self.assertAlmostEqual(q[1], sin)

    def test_vector_space(self):
        """Check dimensions of regular tangent space."""
        robot = load_robot_description(
            "upkie_description", root_joint=kin.JointModelFreeFlyer()
        )
        nv = robot.model.nv
        tangent = VectorSpace(robot.model.nv)
        self.assertEqual(tangent.eye.shape, (nv, nv))
        self.assertEqual(tangent.ones.shape, (nv,))
        self.assertEqual(tangent.zeros.shape, (nv,))
