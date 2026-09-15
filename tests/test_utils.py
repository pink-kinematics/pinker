# SPDX-License-Identifier: Apache-2.0

"""Test fixture for other library features."""

import unittest

from pinker import load_robot_description
from pinker import kinematics as kin
from pinker.utils import VectorSpace


class TestUtils(unittest.TestCase):
    """Test utility classes and functions."""

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
