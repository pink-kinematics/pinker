# SPDX-License-Identifier: Apache-2.0

"""Lie-algebra operations: internal consistency and Pinocchio equivalence."""

import unittest

import numpy as np
import pinocchio as pin

from pinker import kinematics as kin


def random_se3(rng: np.random.Generator) -> kin.SE3:
    """Random rigid transform."""
    w = rng.standard_normal(3)
    return kin.SE3(kin.exp3(w), rng.standard_normal(3))


class TestLie(unittest.TestCase):
    """Test Lie-group operations of the kinematics backend."""

    def test_exp_log_roundtrip(self):
        """exp6 and log6 are inverse of each other."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            nu = rng.standard_normal(6)
            # log(exp(nu)) == nu only holds for rotations below pi
            angle = np.linalg.norm(nu[3:])
            if angle >= np.pi:
                nu[3:] *= 0.9 * np.pi / angle
            M = kin.exp6(nu)
            self.assertTrue(np.allclose(kin.log(M).vector, nu, atol=1e-10))

    def test_log_matches_pinocchio(self):
        """log6 and Jlog6 match Pinocchio, including near-pi rotations."""
        rng = np.random.default_rng(1)
        for _ in range(100):
            M = random_se3(rng)
            M_pin = pin.SE3(M.rotation, M.translation)
            self.assertTrue(
                np.allclose(
                    kin.log(M).vector, pin.log(M_pin).vector, atol=1e-10
                )
            )
            self.assertTrue(
                np.allclose(kin.Jlog6(M), pin.Jlog6(M_pin), atol=1e-10)
            )
        for _ in range(100):
            axis = rng.standard_normal(3)
            axis /= np.linalg.norm(axis)
            for theta in (np.pi, np.pi - 1e-7, np.pi - 1e-3, -np.pi + 1e-5):
                with self.subTest(theta=theta):
                    R = kin.exp3(axis * theta)
                    p = rng.standard_normal(3)
                    M = kin.SE3(R, p)
                    M_pin = pin.SE3(R, p)
                    self.assertTrue(
                        np.allclose(
                            kin.log(M).vector,
                            pin.log(M_pin).vector,
                            atol=1e-6,
                        )
                    )

    def test_jlog6_finite_difference(self):
        """Jlog6 matches a finite-difference approximation.

        Pinocchio's Jlog6 is the right Jacobian: for a perturbation delta of
        M in its local frame, log(M exp(delta)) = log(M) + Jlog6(M) delta.
        """
        rng = np.random.default_rng(2)
        eps = 1e-7
        for _ in range(20):
            M = random_se3(rng)
            J = kin.Jlog6(M)
            J_fd = np.empty((6, 6))
            log_M = kin.log(M).vector
            for k in range(6):
                delta = np.zeros(6)
                delta[k] = eps
                M_pert = M * kin.exp6(delta)
                J_fd[:, k] = (kin.log(M_pert).vector - log_M) / eps
            self.assertTrue(np.allclose(J, J_fd, atol=1e-5))

    def test_se3_actions(self):
        """SE3 composition, inverse and adjoint are consistent."""
        rng = np.random.default_rng(3)
        for _ in range(20):
            A, B = random_se3(rng), random_se3(rng)
            AB = A * B
            self.assertTrue(
                np.allclose(
                    AB.homogeneous,
                    A.homogeneous @ B.homogeneous,
                    atol=1e-12,
                )
            )
            self.assertTrue(
                np.allclose(
                    A.actInv(B).homogeneous,
                    A.inverse().homogeneous @ B.homogeneous,
                    atol=1e-12,
                )
            )
            self.assertTrue(
                np.allclose(
                    (A.inverse() * A).homogeneous, np.eye(4), atol=1e-12
                )
            )
            point = rng.standard_normal(3)
            self.assertTrue(
                np.allclose(
                    A.act(point),
                    A.rotation @ point + A.translation,
                    atol=1e-12,
                )
            )
            # Adjoint: Ad_A nu twists transform like A nu A^-1
            A_pin = pin.SE3(A.rotation, A.translation)
            self.assertTrue(np.allclose(A.action, A_pin.action, atol=1e-12))
            self.assertTrue(
                np.allclose(A.actionInverse, A_pin.actionInverse, atol=1e-12)
            )

    def test_skew(self):
        """skew(v) w equals the cross product of v and w."""
        rng = np.random.default_rng(4)
        for _ in range(10):
            v, w = rng.standard_normal(3), rng.standard_normal(3)
            self.assertTrue(
                np.allclose(kin.skew(v) @ w, np.cross(v, w), atol=1e-12)
            )
