# SPDX-License-Identifier: Apache-2.0

"""Planar rotations: internal consistency and Pinocchio equivalence."""

import unittest

import numpy as np
import pinocchio as pin

from pinker import kinematics as kin


class TestSO2(unittest.TestCase):
    """Test SO(2) exponential and logarithm maps."""

    def setUp(self):
        """Seed the angles drawn by each test."""
        np.random.seed(42)

    def test_exp_so2_is_a_rotation(self):
        """exp_so2 yields orthogonal matrices of determinant one."""
        for theta in np.random.uniform(-10.0, 10.0, 20):
            R = kin.exp_so2(theta)
            self.assertTrue(np.allclose(R @ R.T, np.eye(2), atol=1e-12))
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)

    def test_exp_log_roundtrip(self):
        """exp_so2 and log_so2 are inverse of each other in [-pi, pi]."""
        for theta in [*np.random.uniform(-np.pi, np.pi, 100), 0.0, 1e-9]:
            self.assertAlmostEqual(
                kin.log_so2(kin.exp_so2(theta)), theta, places=10
            )
        self.assertAlmostEqual(abs(kin.log_so2(kin.exp_so2(np.pi))), np.pi)

    def test_log_wraps_angles(self):
        """log_so2 wraps angles to [-pi, pi]."""
        for theta in np.random.uniform(-10.0, 10.0, 100):
            wrapped = np.arctan2(np.sin(theta), np.cos(theta))
            self.assertAlmostEqual(
                kin.log_so2(kin.exp_so2(theta)), wrapped, places=10
            )

    def test_matches_pinocchio(self):
        """exp_so2 and log_so2 match Pinocchio's SO2 Lie group."""
        so2 = pin.liegroups.SO2()
        for theta in [*np.random.uniform(-np.pi, np.pi, 100), 0.0, np.pi]:
            c, s = so2.integrate(so2.neutral, np.array([theta]))
            self.assertTrue(
                np.allclose(kin.exp_so2(theta), [[c, -s], [s, c]], atol=1e-12)
            )
            R = kin.exp_so2(theta)
            log_pin = so2.difference(so2.neutral, R[:, 0])
            self.assertAlmostEqual(kin.log_so2(R), log_pin[0], places=12)
