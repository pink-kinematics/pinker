# SPDX-License-Identifier: Apache-2.0

"""Tests for the URDF visuals parser."""

import os
import tempfile
import unittest

import numpy as np

from pinker import kinematics as kin

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


class TestUrdf(unittest.TestCase):
    """Test the URDF parser."""

    def test_visuals_parser(self):
        """URDF visuals are parsed with shapes, colors and placements."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            urdf_path = os.path.join(tmp_dir, "visual_test.urdf")
            with open(urdf_path, "w", encoding="utf-8") as urdf_file:
                urdf_file.write(VISUAL_URDF)
            model = kin.buildModelFromUrdf(urdf_path)
            geometry = kin.buildGeomFromUrdf(model, urdf_path)
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
