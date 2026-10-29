# SPDX-License-Identifier: Apache-2.0

"""Tests for the MJCF parser."""

import os
import tempfile
import unittest

import numpy as np

from pinker import kinematics as kin
from pinker.kinematics.geometry import DEFAULT_COLOR

TEST_MJCF = """
<mujoco model="mjcf_test">
  <compiler angle="degree" meshdir="assets"/>
  <default>
    <joint axis="0 1 0"/>
    <default class="arm">
      <joint range="-90 90" actuatorfrcrange="-5 5"/>
      <default class="wrist">
        <joint axis="1 0 0"/>
      </default>
    </default>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="2"/>
    </default>
    <default class="collision">
      <geom group="3"/>
    </default>
  </default>
  <asset>
    <mesh file="part.stl" scale="2 2 2"/>
    <material name="red" rgba="1 0 0 1"/>
  </asset>
  <worldbody>
    <body name="base" pos="0 0 1">
      <inertial pos="0 0 0.1" mass="2" diaginertia="0.1 0.2 0.3"/>
      <geom type="box" size="0.1 0.2 0.3" class="visual" material="red"/>
      <geom type="box" size="0.1 0.2 0.3" class="collision"/>
      <body name="upper_arm" pos="0 0 0.1" euler="0 0 90" childclass="arm">
        <joint name="shoulder"/>
        <inertial pos="0 0 0.2" mass="1"
          fullinertia="0.01 0.02 0.03 0.001 0.002 0.003"/>
        <geom type="mesh" mesh="part" class="visual"/>
        <site name="elbow_site" pos="0 0 0.4"/>
        <body name="forearm" pos="0 0 0.4" axisangle="0 0 1 90">
          <joint name="elbow" class="wrist" pos="0 0 -0.05"/>
          <inertial pos="0 0 0.1" mass="0.5" diaginertia="0.001 0.001 0.001"/>
          <body name="tool" pos="0 0 0.3" xyaxes="0 1 0 -1 0 0">
            <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
            <geom type="cylinder" size="0.01 0.05" rgba="0 1 0 1"/>
            <geom type="cylinder" fromto="0 0 0 0 0 0.2" size="0.01"/>
            <site name="tool_tip" pos="0 0 0.05" zaxis="1 0 0"/>
          </body>
        </body>
      </body>
      <body name="slider" pos="0.5 0 0">
        <joint name="rail" type="slide" axis="1 0 0" range="-0.2 0.3"/>
        <inertial pos="0 0 0" quat="0.7071068 0 0 0.7071068" mass="1"
          diaginertia="0.001 0.002 0.003"/>
      </body>
      <body name="ball_body" pos="0 0.5 0">
        <joint type="ball"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.001 0.001 0.001"/>
      </body>
      <body name="spinner" pos="0 -0.5 0">
        <joint name="spin"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.001 0.001 0.001"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

FREE_MJCF = """
<mujoco model="free_test">
  <worldbody>
    <body name="floating" pos="0 0 0.5" quat="0 0 0 1">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <body name="leg" pos="0 0 -0.2">
        <joint name="hip" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class TestMjcf(unittest.TestCase):
    """Test the MJCF parser."""

    def setUp(self):
        """Build the test model."""
        self.model = kin.build_model_from_mjcf_xml(TEST_MJCF)

    def body_placement(
        self, body_attributes: str, compiler: str = ""
    ) -> kin.SE3:
        """Placement of a single top-level body with given attributes.

        Args:
            body_attributes: Attributes of the body element.
            compiler: Attributes of the compiler element.

        Returns:
            Placement of the BODY frame of the body.
        """
        model = kin.build_model_from_mjcf_xml(
            f"<mujoco><compiler {compiler}/><worldbody>"
            f'<body name="b" {body_attributes}/>'
            "</worldbody></mujoco>"
        )
        return model.frames[model.get_frame_id("b")].placement

    def test_dimensions(self):
        """Hinge and slide joints have one coordinate, ball joints four."""
        self.assertEqual(self.model.name, "mjcf_test")
        self.assertEqual(self.model.nq, 8)
        self.assertEqual(self.model.nv, 7)

    def test_joint_order(self):
        """Joints are added depth-first in file order, unnamed ones named."""
        self.assertEqual(
            list(self.model.names),
            [
                "universe",
                "shoulder",
                "elbow",
                "rail",
                "ball_bodyJoint_0",
                "spin",
            ],
        )
        self.assertEqual(list(self.model.parents), [0, 0, 1, 0, 0, 0])

    def test_default_classes(self):
        """Joint axes come from default classes and their parent classes."""
        self.assertEqual(
            [joint.shortname() for joint in self.model.joints[1:]],
            [
                "JointModelRY",  # top-level default, through childclass
                "JointModelRX",  # nested wrist class
                "JointModelPX",
                "JointModelSpherical",
                "JointModelRY",  # top-level default
            ],
        )

    def test_configuration_limits(self):
        """Hinge ranges are converted to radians, not slide ranges."""
        quarter = np.pi / 2
        inf = np.inf
        self.assertTrue(
            np.allclose(
                self.model.lower_position_limit,
                [-quarter, -quarter, -0.2, -1.01, -1.01, -1.01, -1.01, -inf],
            )
        )
        self.assertTrue(
            np.allclose(
                self.model.upper_position_limit,
                [quarter, quarter, 0.3, 1.01, 1.01, 1.01, 1.01, inf],
            )
        )

    def test_velocity_and_effort_limits(self):
        """Efforts come from actuatorfrcrange, velocities are unbounded."""
        inf = np.inf
        self.assertTrue(
            np.allclose(
                self.model.effort_limit, [5.0, 5.0, inf, inf, inf, inf, inf]
            )
        )
        self.assertTrue(np.all(np.isinf(self.model.velocity_limit)))

    def test_joint_placements(self):
        """Joint placements compose body poses and joint positions."""
        R_z90 = kin.rotation_from_axis_angle([0.0, 0.0, 1.0], np.deg2rad(90.0))
        shoulder = self.model.joint_placements[1]
        self.assertTrue(np.allclose(shoulder.rotation, R_z90))
        self.assertTrue(np.allclose(shoulder.translation, [0.0, 0.0, 1.1]))
        elbow = self.model.joint_placements[2]
        self.assertTrue(np.allclose(elbow.rotation, R_z90))
        self.assertTrue(np.allclose(elbow.translation, [0.0, 0.0, 0.35]))
        rail = self.model.joint_placements[3]
        self.assertTrue(np.allclose(rail.rotation, np.eye(3)))
        self.assertTrue(np.allclose(rail.translation, [0.5, 0.0, 1.0]))

    def test_frames(self):
        """Bodies, joints and sites add frames in Pinocchio's order."""
        FT = kin.FrameType
        self.assertEqual(
            [(frame.name, frame.type) for frame in self.model.frames[1:]],
            [
                ("base", FT.BODY),
                ("shoulder", FT.JOINT),
                ("upper_arm", FT.BODY),
                ("elbow_site", FT.OP_FRAME),
                ("elbow", FT.JOINT),
                ("forearm", FT.BODY),
                ("tool_fixed", FT.FIXED_JOINT),
                ("tool", FT.BODY),
                ("tool_tip", FT.OP_FRAME),
                ("rail", FT.JOINT),
                ("slider", FT.BODY),
                ("ball_bodyJoint_0", FT.JOINT),
                ("ball_body", FT.BODY),
                ("spin", FT.JOINT),
                ("spinner", FT.BODY),
            ],
        )

    def test_fixed_bodies_attach_to_their_supporting_joint(self):
        """Bodies without joint and sites hang off the nearest joint."""
        R_y90 = kin.rotation_from_axis_angle([0.0, 1.0, 0.0], np.deg2rad(90.0))
        R_z90 = kin.rotation_from_axis_angle([0.0, 0.0, 1.0], np.deg2rad(90.0))
        elbow_id = self.model.get_joint_id("elbow")
        tool = self.model.frames[self.model.get_frame_id("tool")]
        self.assertEqual(tool.parent_joint, elbow_id)
        self.assertTrue(np.allclose(tool.placement.rotation, R_z90))
        self.assertTrue(np.allclose(tool.placement.translation, [0, 0, 0.35]))
        tip = self.model.frames[self.model.get_frame_id("tool_tip")]
        self.assertEqual(tip.parent_joint, elbow_id)
        self.assertEqual(tip.parent_frame, self.model.get_frame_id("tool"))
        self.assertTrue(np.allclose(tip.placement.rotation, R_z90 @ R_y90))
        self.assertTrue(np.allclose(tip.placement.translation, [0, 0, 0.4]))

    def test_inertias(self):
        """Inertias are rotated, displaced and merged into their joints."""
        base = self.model.inertias[0]
        self.assertAlmostEqual(base.mass, 2.0)
        self.assertTrue(np.allclose(base.lever, [0.0, 0.0, 1.1]))
        shoulder = self.model.inertias[1]
        self.assertTrue(
            np.allclose(
                shoulder.inertia,
                [
                    [0.01, 0.001, 0.002],
                    [0.001, 0.02, 0.003],
                    [0.002, 0.003, 0.03],
                ],
            )
        )
        elbow = self.model.inertias[2]  # forearm with the tool merged in
        self.assertAlmostEqual(elbow.mass, 0.6)
        lever_z = (0.5 * 0.15 + 0.1 * 0.35) / 0.6
        self.assertTrue(np.allclose(elbow.lever, [0.0, 0.0, lever_z]))
        rail = self.model.inertias[3]
        self.assertTrue(
            np.allclose(rail.inertia, np.diag([0.002, 0.001, 0.003]))
        )

    def test_orientations(self):
        """All orientation specifiers give the expected rotations."""
        quarter, half = np.deg2rad(90.0), np.deg2rad(180.0)
        R_x90 = kin.rotation_from_axis_angle([1.0, 0.0, 0.0], quarter)
        R_y90 = kin.rotation_from_axis_angle([0.0, 1.0, 0.0], quarter)
        R_z90 = kin.rotation_from_axis_angle([0.0, 0.0, 1.0], quarter)
        R_x180 = kin.rotation_from_axis_angle([1.0, 0.0, 0.0], half)
        cases = [
            ('quat="2 0 0 0"', "", np.eye(3)),
            ('quat="1 0 0 1"', "", R_z90),
            ('axisangle="0 0 2 90"', "", R_z90),
            ('axisangle="0 0 1 1.5707963"', 'angle="radian"', R_z90),
            ('euler="90 90 0"', "", R_x90 @ R_y90),
            ('euler="90 90 0"', 'eulerseq="XYZ"', R_y90 @ R_x90),
            ('euler="90 0 0"', 'eulerseq="zyx"', R_z90),
            ('xyaxes="0 1 0 -1 0.5 0"', "", R_z90),
            ('zaxis="1 0 0"', "", R_y90),
            ('zaxis="0 0 3"', "", np.eye(3)),
            ('zaxis="0 0 -1"', "", R_x180),
        ]
        for body_attributes, compiler, expected in cases:
            with self.subTest(body=body_attributes, compiler=compiler):
                placement = self.body_placement(body_attributes, compiler)
                self.assertTrue(
                    np.allclose(placement.rotation, expected, atol=1e-7)
                )
        placement = self.body_placement('pos="1 2 3"')
        self.assertTrue(np.allclose(placement.translation, [1, 2, 3]))

    def test_root_joint(self):
        """A root joint connects a fixed root body to the world."""
        model = kin.build_model_from_mjcf_xml(
            TEST_MJCF, kin.JointModelPlanar()
        )
        self.assertEqual(model.names[1], "root_joint")
        self.assertEqual(model.nq, self.model.nq + 4)
        self.assertTrue(
            model.joint_placements[1].is_approx(kin.SE3.Identity())
        )
        base = model.frames[model.get_frame_id("base")]
        self.assertEqual(base.parent_joint, 1)

    def test_free_joint(self):
        """Free joints coincide with the world and have unbounded positions."""
        model = kin.build_model_from_mjcf_xml(FREE_MJCF)
        self.assertEqual(model.names[1], "floating_free")
        self.assertEqual(model.joints[1].shortname(), "JointModelFreeFlyer")
        self.assertTrue(
            model.joint_placements[1].is_approx(kin.SE3.Identity())
        )
        self.assertTrue(np.all(np.isinf(model.upper_position_limit[:3])))
        self.assertTrue(np.allclose(model.upper_position_limit[3:7], 1.01))
        self.assertTrue(np.allclose(model.lower_position_limit[3:7], -1.01))
        hip = model.joint_placements[model.get_joint_id("hip")]
        self.assertTrue(np.allclose(hip.translation, [0.0, 0.0, -0.2]))

    def test_root_joint_ignored_with_free_joint(self):
        """A root joint is ignored, with a warning, if there is a freejoint."""
        with self.assertLogs(level="WARNING") as logs:
            model = kin.build_model_from_mjcf_xml(
                FREE_MJCF, kin.JointModelFreeFlyer()
            )
        self.assertIn("floating_free", logs.output[0])
        self.assertEqual(model.nq, 7 + 1)
        self.assertEqual(model.names[1], "floating_free")

    def test_unsupported_features(self):
        """Unsupported MJCF features raise rather than being skipped."""
        cases = {
            "several joints": '<body name="b"><joint/><joint/></body>',
            "inertia from geoms": '<body name="b"><geom size="0.1"/></body>',
            "nested free joint": (
                '<body name="a"><body name="b"><freejoint/></body></body>'
            ),
            "joint reference": '<body name="b"><joint ref="0.1"/></body>',
            "unknown class": '<body name="b"><joint class="nope"/></body>',
            "two orientations": (
                '<body name="b" quat="1 0 0 0" euler="0 1 0"/>'
            ),
            "frame": '<body name="b"><frame><body name="c"/></frame></body>',
        }
        for feature, worldbody in cases.items():
            with self.subTest(feature=feature):
                with self.assertRaises(ValueError):
                    kin.build_model_from_mjcf_xml(
                        f"<mujoco><worldbody>{worldbody}</worldbody></mujoco>"
                    )
        with self.assertRaises(ValueError):
            kin.build_model_from_mjcf_xml(
                '<mujoco><include file="robot.xml"/></mujoco>'
            )
        with self.assertRaises(ValueError):
            kin.build_model_from_mjcf_xml(
                '<mujoco><compiler inertiafromgeom="true"/></mujoco>'
            )
        with self.assertRaises(ValueError):
            kin.build_model_from_mjcf_xml('<robot name="urdf"/>')
        with self.assertRaises(ValueError):  # which body gets the root joint?
            kin.build_model_from_mjcf_xml(
                '<mujoco><worldbody><body name="a"/><body name="b"/>'
                "</worldbody></mujoco>",
                kin.JointModelFreeFlyer(),
            )

    def test_visuals_parser(self):
        """Visual geoms are parsed with shapes, colors and placements."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mjcf_path = os.path.join(tmp_dir, "mjcf_test.xml")
            with open(mjcf_path, "w", encoding="utf-8") as mjcf_file:
                mjcf_file.write(TEST_MJCF)
            os.mkdir(os.path.join(tmp_dir, "assets"))
            stl_path = os.path.join(tmp_dir, "assets", "part.stl")
            open(stl_path, "w", encoding="utf-8").close()
            model = kin.build_model_from_mjcf(mjcf_path)
            geometry = kin.build_geom_from_mjcf(model, mjcf_path)
            os.remove(stl_path)
            without_mesh = kin.build_geom_from_mjcf(model, mjcf_path)
        self.assertEqual(
            [obj.name for obj in geometry.geometry_objects],
            ["base_0", "upper_arm_0", "tool_0", "tool_1"],  # base_1 collides
        )
        box, mesh, cylinder, fromto = geometry.geometry_objects
        self.assertEqual(box.shape, "box")
        self.assertTrue(np.allclose(box.size, [0.2, 0.4, 0.6]))
        self.assertTrue(np.allclose(box.mesh_color, [1.0, 0.0, 0.0, 1.0]))
        self.assertEqual(box.parent_joint, 0)
        self.assertTrue(np.allclose(box.placement.translation, [0, 0, 1]))
        self.assertEqual(mesh.shape, "mesh")
        self.assertEqual(mesh.mesh_path, stl_path)
        self.assertTrue(np.allclose(mesh.mesh_scale, [2.0, 2.0, 2.0]))
        self.assertTrue(np.allclose(mesh.mesh_color, DEFAULT_COLOR))
        self.assertEqual(mesh.parent_joint, model.get_joint_id("shoulder"))
        self.assertEqual(cylinder.shape, "cylinder")
        self.assertTrue(np.allclose(cylinder.size, [0.01, 0.1]))
        self.assertTrue(np.allclose(cylinder.mesh_color, [0, 1, 0, 1]))
        tool = model.frames[model.get_frame_id("tool")]
        self.assertEqual(cylinder.parent_joint, tool.parent_joint)
        self.assertTrue(cylinder.placement.is_approx(tool.placement))
        self.assertTrue(np.allclose(fromto.size, [0.01, 0.2]))
        self.assertTrue(
            fromto.placement.is_approx(
                tool.placement * kin.SE3(np.eye(3), np.array([0, 0, 0.1]))
            )
        )
        self.assertEqual(without_mesh.ngeoms, 3)  # missing mesh file skipped


if __name__ == "__main__":
    unittest.main()
