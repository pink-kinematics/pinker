# SPDX-License-Identifier: Apache-2.0

"""Tests for configuration-vector helpers."""

import unittest

import numpy as np

from pinker import kinematics as kin

# Two bounded joints and one continuous joint, which is the case that
# distinguishes configuration from tangent coordinates: a continuous joint
# stores (cos, sin) of its angle, so nq = 2 while nv = 1.
DUMMY_URDF = """<?xml version="1.0"?>
<robot name="dummy">
  <link name="base"/>
  <link name="thigh"/>
  <link name="shank"/>
  <link name="foot"/>
  <joint name="hip" type="revolute">
    <parent link="base"/><child link="thigh"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="knee" type="revolute">
    <parent link="thigh"/><child link="shank"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="wheel" type="continuous">
    <parent link="shank"/><child link="foot"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
"""


class TestCustomConfiguration(unittest.TestCase):
    """Test the custom_configuration helper."""

    def setUp(self):
        """Build the dummy model."""
        self.model = kin.build_model_from_xml(DUMMY_URDF)

    def idx_q(self, joint_name: str) -> int:
        """Index of a joint in the configuration vector.

        Args:
            joint_name: Name of the joint.

        Returns:
            Index of the first configuration coordinate of the joint.
        """
        return self.model.joints[self.model.get_joint_id(joint_name)].idx_q

    def test_named_joints_take_their_value(self):
        """Named joints get their value, others their neutral one."""
        q = kin.custom_configuration(self.model, hip=0.2, knee=-0.2)
        self.assertAlmostEqual(q[self.idx_q("hip")], 0.2)
        self.assertAlmostEqual(q[self.idx_q("knee")], -0.2)
        # The continuous joint keeps its neutral value (cos, sin) = (1, 0)
        wheel = self.idx_q("wheel")
        self.assertAlmostEqual(q[wheel], 1.0)
        self.assertAlmostEqual(q[wheel + 1], 0.0)

    def test_unbounded_joint_takes_cos_sin(self):
        """A continuous joint is set from its (cos, sin) pair."""
        cos, sin = np.sqrt(3) / 2.0, 0.5
        q = kin.custom_configuration(self.model, wheel=[cos, sin])
        wheel = self.idx_q("wheel")
        self.assertAlmostEqual(q[wheel], cos)
        self.assertAlmostEqual(q[wheel + 1], sin)

    def test_single_number_for_unbounded_joint_raises(self):
        """A single number for a continuous joint yields an error."""
        with self.assertRaises(ValueError):
            kin.custom_configuration(self.model, wheel=0.0)

    def test_unknown_joint_raises(self):
        """Naming a joint the model does not have yields an error."""
        with self.assertRaises(ValueError):
            kin.custom_configuration(self.model, no_such_joint=0.0)
