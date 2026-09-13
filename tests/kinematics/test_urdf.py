# SPDX-License-Identifier: Apache-2.0

"""Tests for the URDF parser."""

import os
import tempfile
import unittest

import numpy as np

from pinker import kinematics as kin

HERE = os.path.dirname(__file__)
WHEELED_URDF = os.path.join(HERE, "wheeled.urdf")
DBL_MAX = np.finfo(np.float64).max

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
            model = kin.build_model_from_urdf(urdf_path)
            geometry = kin.build_geom_from_urdf(model, urdf_path)
        self.assertEqual(geometry.ngeoms, 3)
        box, sphere, cylinder = geometry.geometry_objects
        self.assertEqual(box.shape, "box")
        self.assertTrue(np.allclose(box.size, [0.3, 0.2, 0.1]))
        self.assertTrue(np.allclose(box.mesh_color, [0.1, 0.2, 0.3, 1.0]))
        self.assertTrue(
            np.allclose(box.placement.translation, [0.1, 0.0, 0.2])
        )
        self.assertEqual(sphere.shape, "sphere")
        self.assertTrue(np.allclose(sphere.mesh_color, [1.0, 0.0, 0.0, 0.5]))
        self.assertEqual(cylinder.shape, "cylinder")
        self.assertEqual(cylinder.parent_joint, model.get_joint_id("hinge"))


class TestWheeledModel(unittest.TestCase):
    """Test the parser on a model with a free-flyer root.

    The wheeled robot has joint types whose configuration and tangent
    dimensions differ: a free-flyer root (nq=7, nv=6) and two continuous joints
    (nq=2, nv=1), along with prismatic and revolute joints on unaligned axes
    and two fixed joints.
    """

    def setUp(self):
        """Load the model with a free-flyer root joint."""
        self.model = kin.build_model_from_urdf(
            WHEELED_URDF, kin.JointModelFreeFlyer()
        )

    def joint(self, name: str) -> kin.JointModel:
        """Joint model of the joint with a given name.

        Args:
            name: Name of the joint.

        Returns:
            Corresponding joint model.
        """
        return self.model.joints[self.model.get_joint_id(name)]

    def test_dimensions(self):
        """Dimensions account for the free-flyer and continuous joints."""
        # 7 + 2 + 1 + 1 + 2 configuration coordinates
        self.assertEqual(self.model.nq, 13)
        # 6 + 1 + 1 + 1 + 1 tangent coordinates
        self.assertEqual(self.model.nv, 10)
        self.assertEqual(self.model.njoints, 6)  # universe included

    def test_joint_order(self):
        """Joints follow a depth-first traversal, children in name order."""
        self.assertEqual(
            list(self.model.names),
            [
                "universe",
                "root_joint",
                "left_wheel_joint",
                "slider_joint",  # under the fixed mast_joint
                "head_joint",
                "right_wheel_joint",
            ],
        )

    def test_joint_shortnames(self):
        """Shortnames report joint type and axis, as in Pinocchio."""
        self.assertEqual(
            self.joint("root_joint").shortname(), "JointModelFreeFlyer"
        )
        # A continuous joint about +z is axis-aligned, one about -z is not
        self.assertEqual(
            self.joint("left_wheel_joint").shortname(), "JointModelRUBZ"
        )
        self.assertEqual(
            self.joint("right_wheel_joint").shortname(),
            "JointModelRevoluteUnboundedUnaligned",
        )
        self.assertEqual(
            self.joint("slider_joint").shortname(),
            "JointModelPrismaticUnaligned",
        )
        self.assertEqual(
            self.joint("head_joint").shortname(), "JointModelRevoluteUnaligned"
        )

    def test_continuous_joints_store_cos_sin(self):
        """Continuous joints take two configuration coordinates for one DoF."""
        for name in ("left_wheel_joint", "right_wheel_joint"):
            with self.subTest(joint=name):
                joint = self.joint(name)
                self.assertEqual((joint.nq, joint.nv), (2, 1))
                q = kin.neutral(self.model)
                cos, sin = q[joint.idx_q : joint.idx_q + 2]
                self.assertAlmostEqual(cos, 1.0)
                self.assertAlmostEqual(sin, 0.0)

    def test_free_flyer_root(self):
        """The root joint leads the configuration and tangent vectors."""
        root = self.joint("root_joint")
        self.assertEqual((root.nq, root.nv), (7, 6))
        self.assertEqual((root.idx_q, root.idx_v), (0, 0))
        q = kin.neutral(self.model)
        # Position, then a unit quaternion with its vector part first
        self.assertTrue(np.allclose(q[:7], [0, 0, 0, 0, 0, 0, 1]))

    def test_configuration_limits(self):
        """Configuration bounds come from the URDF, with Pinocchio's quirks."""
        lower = self.model.lower_position_limit
        upper = self.model.upper_position_limit

        # Bounded joints get the values of their <limit> tag
        slider = self.joint("slider_joint")
        self.assertAlmostEqual(lower[slider.idx_q], -0.5)
        self.assertAlmostEqual(upper[slider.idx_q], 0.7)
        head = self.joint("head_joint")
        self.assertAlmostEqual(lower[head.idx_q], -1.5)
        self.assertAlmostEqual(upper[head.idx_q], 1.5)

        # Continuous joints bound their (cos, sin) pair by 1.01
        wheel = self.joint("left_wheel_joint")
        self.assertTrue(
            np.allclose(lower[wheel.idx_q : wheel.idx_q + 2], -1.01)
        )
        self.assertTrue(
            np.allclose(upper[wheel.idx_q : wheel.idx_q + 2], 1.01)
        )

        # A root joint added programmatically defaults to the largest double,
        # not to infinity
        self.assertTrue(np.allclose(lower[:7], -DBL_MAX))
        self.assertTrue(np.allclose(upper[:7], DBL_MAX))

    def test_which_coordinates_have_a_configuration_limit(self):
        """Only bounded coordinates are flagged as having a limit."""
        flags = self.model.has_configuration_limit()
        self.assertEqual(flags.shape, (self.model.nq,))
        # Free-flyer: its position is bounded, its quaternion is not
        self.assertTrue(np.all(flags[:3]))
        self.assertFalse(np.any(flags[3:7]))
        for name in ("left_wheel_joint", "right_wheel_joint"):
            with self.subTest(joint=name):
                joint = self.joint(name)
                self.assertFalse(
                    np.any(flags[joint.idx_q : joint.idx_q + joint.nq])
                )
        for name in ("slider_joint", "head_joint"):
            with self.subTest(joint=name):
                self.assertTrue(flags[self.joint(name).idx_q])

    def test_velocity_and_effort_limits(self):
        """Velocity and effort bounds come from the URDF, per DoF."""
        velocity = self.model.velocity_limit
        effort = self.model.effort_limit
        expected = {
            "left_wheel_joint": (8.0, 10.0),
            "right_wheel_joint": (8.0, 10.0),
            "slider_joint": (1.5, 20.0),
            "head_joint": (3.0, 5.0),
        }
        for name, (max_velocity, max_effort) in expected.items():
            with self.subTest(joint=name):
                idx_v = self.joint(name).idx_v
                self.assertAlmostEqual(velocity[idx_v], max_velocity)
                self.assertAlmostEqual(effort[idx_v], max_effort)

    def test_frames(self):
        """Each joint adds a JOINT frame, each link a BODY frame."""
        frames = {frame.name: frame for frame in self.model.frames}
        for name in self.model.names[1:]:  # skip the universe
            with self.subTest(frame=name):
                self.assertEqual(frames[name].type, kin.FrameType.JOINT)
        for link in ("base", "left_wheel", "right_wheel", "mast", "slider"):
            with self.subTest(frame=link):
                self.assertEqual(frames[link].type, kin.FrameType.BODY)

    def test_fixed_joints_attach_to_their_supporting_joint(self):
        """Fixed joints become frames of the closest movable ancestor."""
        frames = {frame.name: frame for frame in self.model.frames}
        for name in ("mast_joint", "antenna_joint"):
            with self.subTest(frame=name):
                self.assertEqual(frames[name].type, kin.FrameType.FIXED_JOINT)
        # The mast hangs from the root joint, the antenna from the head
        self.assertEqual(
            frames["mast_joint"].parent_joint,
            self.model.get_joint_id("root_joint"),
        )
        self.assertEqual(
            frames["antenna_joint"].parent_joint,
            self.model.get_joint_id("head_joint"),
        )
        # Their child links follow the same joint
        self.assertEqual(
            frames["antenna"].parent_joint,
            self.model.get_joint_id("head_joint"),
        )

    def test_inertias(self):
        """Link inertias are read, fixed-joint ones merged into their joint."""
        masses = {
            name: self.model.inertias[self.model.get_joint_id(name)].mass
            for name in self.model.names[1:]
        }
        # The base and the massless mast both sit on the root joint
        self.assertAlmostEqual(masses["root_joint"], 4.2)
        self.assertAlmostEqual(masses["left_wheel_joint"], 0.6)
        self.assertAlmostEqual(masses["slider_joint"], 1.1)
        self.assertAlmostEqual(
            masses["head_joint"], 0.3
        )  # antenna is massless
        total = sum(inertia.mass for inertia in self.model.inertias)
        self.assertAlmostEqual(total, 6.8)
