# SPDX-License-Identifier: Apache-2.0

"""Rotations: internal consistency and Pinocchio equivalence."""

import unittest

import numpy as np
import pinocchio as pin

from pinker import kinematics as kin


class TestSO3(unittest.TestCase):
    """Test rotation helpers and their Lie-group operations."""

    def setUp(self):
        """Seed the rotations drawn by each test."""
        np.random.seed(42)

    def random_quaternion(self) -> np.ndarray:
        """Random unit quaternion.

        Returns:
            Unit quaternion, as (x, y, z, w).
        """
        quat = np.random.standard_normal(4)
        return quat / np.linalg.norm(quat)

    def test_skew(self):
        """skew(v) w equals the cross product of v and w."""
        for _ in range(10):
            v = np.random.standard_normal(3)
            w = np.random.standard_normal(3)
            self.assertTrue(
                np.allclose(kin.skew(v) @ w, np.cross(v, w), atol=1e-12)
            )

    def test_quaternion_to_matrix_matches_pinocchio(self):
        """Our quaternion conversion agrees with Pinocchio's.

        Ours takes the (x, y, z, w) order in which quaternions are stored,
        while Eigen constructs them scalar first.
        """
        for _ in range(100):
            x, y, z, w = self.random_quaternion()
            self.assertTrue(
                np.allclose(
                    kin.quaternion_to_matrix([x, y, z, w]),
                    pin.Quaternion(w, x, y, z).matrix(),
                    atol=1e-12,
                )
            )

    def test_quaternion_wxyz_inverts_quaternion_to_matrix(self):
        """Going matrix to quaternion and back is the identity."""
        for _ in range(100):
            quat = self.random_quaternion()
            R = kin.quaternion_to_matrix(quat)
            w, x, y, z = kin.quaternion_wxyz(R)
            self.assertTrue(
                np.allclose(
                    kin.quaternion_to_matrix([x, y, z, w]), R, atol=1e-12
                )
            )

    def test_quaternion_wxyz_is_a_unit_quaternion(self):
        """Quaternions read off a rotation matrix have unit norm."""
        for _ in range(100):
            R = kin.exp3(np.random.standard_normal(3))
            self.assertAlmostEqual(
                np.linalg.norm(kin.quaternion_wxyz(R)), 1.0, places=12
            )

    def test_rpy_to_matrix_matches_pinocchio(self):
        """Roll-pitch-yaw angles are composed as in Pinocchio."""
        for _ in range(100):
            roll, pitch, yaw = np.random.uniform(-np.pi, np.pi, 3)
            self.assertTrue(
                np.allclose(
                    kin.rpy_to_matrix(roll, pitch, yaw),
                    pin.rpy.rpyToMatrix(roll, pitch, yaw),
                    atol=1e-12,
                )
            )

    def test_rotation_from_axis_angle_matches_pinocchio(self):
        """Axis-angle rotations agree with Pinocchio's exponential map."""
        for _ in range(100):
            axis = np.random.standard_normal(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(-2.0 * np.pi, 2.0 * np.pi)
            self.assertTrue(
                np.allclose(
                    kin.rotation_from_axis_angle(axis, angle),
                    pin.exp3(axis * angle),
                    atol=1e-12,
                )
            )

    def test_rotation_from_z_axis_matches_eigen(self):
        """Minimal rotation from the z-axis agrees with Eigen's."""
        z_axes = [np.random.standard_normal(3) for _ in range(100)]
        # Close to but not exactly opposite, where the rotation axis is
        # arbitrary and Eigen picks a different one than MuJoCo
        z_axes += [np.array([0.0, 0.0, 1.0]), np.array([1e-3, 0.0, -1.0])]
        for z_axis in z_axes:
            z_axis /= np.linalg.norm(z_axis)
            R = kin.rotation_from_z_axis(z_axis)
            self.assertTrue(
                np.allclose(
                    R,
                    pin.Quaternion.FromTwoVectors(
                        np.array([0.0, 0.0, 1.0]), z_axis
                    ).matrix(),
                    atol=1e-9,
                )
            )

    def test_rotation_from_z_axis_opposite(self):
        """Pointing down the z-axis yields a half turn about the x-axis.

        The axis is arbitrary in this case, we pick the same as MuJoCo.
        """
        R = kin.rotation_from_z_axis(np.array([0.0, 0.0, -1.0]))
        self.assertTrue(np.allclose(R, np.diag([1.0, -1.0, -1.0])))

    def test_exp3_log3_roundtrip(self):
        """exp3 and log3 are inverse of each other below pi."""
        for _ in range(100):
            w = np.random.standard_normal(3)
            angle = np.linalg.norm(w)
            if angle >= np.pi:
                w *= 0.9 * np.pi / angle
            self.assertTrue(np.allclose(kin.log3(kin.exp3(w)), w, atol=1e-10))

    def test_exp3_matches_pinocchio(self):
        """exp3 agrees with Pinocchio, including near-pi rotations."""
        for theta in (0.0, 1e-8, 0.5, np.pi - 1e-3, np.pi):
            with self.subTest(theta=theta):
                axis = np.random.standard_normal(3)
                axis /= np.linalg.norm(axis)
                self.assertTrue(
                    np.allclose(
                        kin.exp3(axis * theta),
                        pin.exp3(axis * theta),
                        atol=1e-10,
                    )
                )
