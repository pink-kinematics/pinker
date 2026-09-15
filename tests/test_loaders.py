# SPDX-License-Identifier: Apache-2.0

"""Test fixture for the robot loader."""

import os
import unittest

from pinker import Configuration, load_robot
from pinker import kinematics as kin

URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "double_pendulum.urdf",
)


class TestLoadRobot(unittest.TestCase):
    """Test that robots load from both URDF files and descriptions."""

    def test_urdf_file(self):
        """A path to a URDF file is loaded from that file."""
        robot = load_robot(URDF_PATH)
        self.assertEqual(robot.nq, 2)
        self.assertEqual(robot.nv, 2)
        self.assertIsNotNone(robot.visual_model)

    def test_description_name(self):
        """A name that is not a path is loaded as a robot description."""
        robot = load_robot("ur3_official_description")
        self.assertEqual(robot.nq, 7)  # wrist_3_joint is continuous
        self.assertEqual(robot.nv, 6)

    def test_configuration_from_robot(self):
        """The loaded robot has what a configuration is built from."""
        robot = load_robot(URDF_PATH)
        configuration = Configuration(robot.model, robot.data, robot.q0)
        self.assertEqual(configuration.q.shape, (robot.nq,))

    def test_root_joint_by_name(self):
        """Root joints can be named by string."""
        robot = load_robot("upkie_description", root_joint="free_flyer")
        self.assertEqual(robot.nq, robot.nv + 1)  # free-flyer quaternion
        self.assertEqual(
            robot.model.joints[1].shortname(), "JointModelFreeFlyer"
        )

    def test_root_joint_by_model(self):
        """Root joints can also be joint models, as in pinker.kinematics."""
        by_name = load_robot("upkie_description", root_joint="free_flyer")
        by_model = load_robot(
            "upkie_description", root_joint=kin.JointModelFreeFlyer()
        )
        self.assertEqual(by_name.nq, by_model.nq)
        self.assertEqual(by_name.nv, by_model.nv)

    def test_unknown_root_joint(self):
        """An unknown root-joint name lists the valid ones."""
        with self.assertRaises(ValueError) as context:
            load_robot("ur3_official_description", root_joint="freeflyer")
        self.assertIn("free_flyer", str(context.exception))

    def test_commit_is_for_descriptions(self):
        """The commit argument does not apply to URDF files."""
        with self.assertRaises(ValueError):
            load_robot(URDF_PATH, commit="0123456789abcdef")

    def test_package_dirs_are_for_urdf_files(self):
        """The package_dirs argument does not apply to descriptions."""
        with self.assertRaises(ValueError):
            load_robot("ur3_official_description", package_dirs=["."])


if __name__ == "__main__":
    unittest.main()
