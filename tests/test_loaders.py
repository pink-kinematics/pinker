# SPDX-License-Identifier: Apache-2.0

"""Test fixture for robot loaders."""

import os
import unittest

import numpy as np

from pinker import (
    Configuration,
    load_robot_description,
    load_robot_mjcf,
    load_robot_urdf,
)
from pinker import kinematics as kin

URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "double_pendulum.urdf",
)

MJCF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "double_pendulum.xml",
)


class TestLoadRobotUrdf(unittest.TestCase):
    """Test that robots load from URDF files."""

    def test_urdf_file(self):
        """A URDF file is loaded with its visuals."""
        robot = load_robot_urdf(URDF_PATH)
        self.assertEqual(robot.nq, 2)
        self.assertEqual(robot.nv, 2)
        self.assertIsNotNone(robot.visual_model)

    def test_configuration_from_robot(self):
        """The loaded robot has what a configuration is built from."""
        robot = load_robot_urdf(URDF_PATH)
        configuration = Configuration(robot.model, robot.data, robot.q0)
        self.assertEqual(configuration.q.shape, (robot.nq,))

    def test_root_joint(self):
        """A root joint adds its own degrees of freedom to the model."""
        robot = load_robot_urdf(URDF_PATH, root_joint="free_flyer")
        self.assertEqual(robot.nq, 2 + 7)
        self.assertEqual(robot.nv, 2 + 6)

    def test_missing_file(self):
        """Loading a file that does not exist raises."""
        with self.assertRaises(FileNotFoundError):
            load_robot_urdf(f"{URDF_PATH}.does_not_exist")

    def test_unknown_root_joint(self):
        """An unknown root-joint name lists the valid ones."""
        with self.assertRaises(ValueError) as context:
            load_robot_urdf(URDF_PATH, root_joint="freeflyer")
        self.assertIn("free_flyer", str(context.exception))


class TestLoadRobotMjcf(unittest.TestCase):
    """Test that robots load from MJCF files."""

    def test_mjcf_file(self):
        """An MJCF file is loaded with its visuals and sites."""
        robot = load_robot_mjcf(MJCF_PATH)
        self.assertEqual(robot.nq, 2)
        self.assertEqual(robot.nv, 2)
        self.assertEqual(robot.visual_model.ngeoms, 3)
        self.assertTrue(robot.model.exist_frame("tip"))

    def test_same_model_as_urdf(self):
        """The MJCF and URDF double pendulums have the same kinematics."""
        mjcf_robot = load_robot_mjcf(MJCF_PATH)
        urdf_robot = load_robot_urdf(URDF_PATH)
        q = np.array([0.3, -0.5])
        mjcf_config = Configuration(mjcf_robot.model, mjcf_robot.data, q)
        urdf_config = Configuration(urdf_robot.model, urdf_robot.data, q)
        mjcf_tip = mjcf_config.get_transform_frame_to_world("tip")
        urdf_tip = urdf_config.get_transform_frame_to_world("link3")
        self.assertTrue(mjcf_tip.is_approx(urdf_tip))

    def test_root_joint(self):
        """A root joint adds its own degrees of freedom to the model."""
        robot = load_robot_mjcf(MJCF_PATH, root_joint="free_flyer")
        self.assertEqual(robot.nq, 2 + 7)
        self.assertEqual(robot.nv, 2 + 6)

    def test_missing_file(self):
        """Loading a file that does not exist raises."""
        with self.assertRaises(FileNotFoundError):
            load_robot_mjcf(f"{MJCF_PATH}.does_not_exist")


class TestLoadRobotDescription(unittest.TestCase):
    """Test that robots load from robot descriptions."""

    def test_description_name(self):
        """A description is loaded by name."""
        robot = load_robot_description("ur3_official_description")
        self.assertEqual(robot.nq, 7)  # wrist_3_joint is continuous
        self.assertEqual(robot.nv, 6)

    def test_root_joint_by_name(self):
        """Root joints can be named by string."""
        robot = load_robot_description(
            "upkie_description", root_joint="free_flyer"
        )
        self.assertEqual(robot.nq, robot.nv + 1)  # free-flyer quaternion
        self.assertEqual(
            robot.model.joints[1].shortname(), "JointModelFreeFlyer"
        )

    def test_root_joint_by_model(self):
        """Root joints can also be joint models, as in pinker.kinematics."""
        by_name = load_robot_description(
            "upkie_description", root_joint="free_flyer"
        )
        by_model = load_robot_description(
            "upkie_description", root_joint=kin.JointModelFreeFlyer()
        )
        self.assertEqual(by_name.nq, by_model.nq)
        self.assertEqual(by_name.nv, by_model.nv)

    def test_unknown_root_joint(self):
        """An unknown root-joint name lists the valid ones."""
        with self.assertRaises(ValueError) as context:
            load_robot_description(
                "ur3_official_description", root_joint="freeflyer"
            )
        self.assertIn("free_flyer", str(context.exception))

    def test_mjcf_description(self):
        """Descriptions with only an MJCF are loaded from it."""
        robot = load_robot_description("go2_mj_description")
        self.assertEqual(robot.nq, 7 + 12)
        self.assertEqual(robot.model.names[1], "base_free")
        self.assertEqual(
            robot.model.joints[1].shortname(), "JointModelFreeFlyer"
        )
        self.assertGreater(robot.visual_model.ngeoms, 0)

    def test_mjcf_description_with_freejoint_ignores_root_joint(self):
        """The freejoint of an MJCF description takes over the root joint."""
        with self.assertLogs(level="WARNING"):
            robot = load_robot_description(
                "go2_mj_description", root_joint="free_flyer"
            )
        self.assertEqual(robot.nq, 7 + 12)

    def test_unknown_description(self):
        """An unknown description name raises."""
        with self.assertRaises(ModuleNotFoundError):
            load_robot_description("no_such_description")


if __name__ == "__main__":
    unittest.main()
