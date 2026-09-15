# SPDX-License-Identifier: Apache-2.0

"""Test fixture for robot loaders."""

import os
import unittest

from pinker import Configuration, load_robot_description, load_robot_urdf
from pinker import kinematics as kin

URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "double_pendulum.urdf",
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

    def test_unknown_description(self):
        """An unknown description name raises."""
        with self.assertRaises(ModuleNotFoundError):
            load_robot_description("no_such_description")


if __name__ == "__main__":
    unittest.main()
