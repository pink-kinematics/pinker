# SPDX-License-Identifier: Apache-2.0

"""Planar rigid transforms: internal consistency and Pinocchio equivalence."""

import unittest

import numpy as np
import pinocchio as pin

from pinker import kinematics as kin


def to_config(M: kin.SE2) -> np.ndarray:
    """Configuration (x, y, cos, sin) of a planar transform in Pinocchio.

    Args:
        M: Planar rigid transform.

    Returns:
        Configuration vector of the transform on Pinocchio's SE2 Lie group.
    """
    return np.array([*M.translation, M.rotation[0, 0], M.rotation[1, 0]])


def to_se3(M: kin.SE2) -> kin.SE3:
    """Rigid transform of a planar transform in the xy-plane.

    Args:
        M: Planar rigid transform.

    Returns:
        Rigid transform rotating about the z-axis, translating in the plane.
    """
    R = np.eye(3)
    R[:2, :2] = M.rotation
    return kin.SE3(R, np.array([*M.translation, 0.0]))


# Indices of (vx, vy, omega_z) in 6D twists (linear, angular)
PLANAR = [0, 1, 5]


class TestSE2(unittest.TestCase):
    """Test SE(2) transforms and their Lie-group operations."""

    def setUp(self):
        """Seed the transforms that SE2.Random draws."""
        np.random.seed(42)

    def test_exp_log_roundtrip(self):
        """exp_se2 and log_se2 are inverse of each other."""
        for _ in range(100):
            nu = np.random.standard_normal(3)
            # log(exp(nu)) == nu only holds for rotations below pi
            if abs(nu[2]) >= np.pi:
                nu[2] *= 0.9 * np.pi / abs(nu[2])
            M = kin.exp_se2(nu)
            self.assertTrue(np.allclose(kin.log_se2(M), nu, atol=1e-10))

    def planar_twists(self):
        """Planar twists, including rotations near zero and pi.

        Returns:
            List of twists (vx, vy, omega).
        """
        omegas = (0.0, 1e-9, 1e-5, 0.5, -2.0, np.pi - 1e-3, np.pi)
        return [np.array([*np.random.standard_normal(2), w]) for w in omegas]

    def planar_transforms(self):
        """Planar transforms, including rotations near zero and pi.

        Returns:
            List of planar rigid transforms.
        """
        transforms = [kin.SE2.Random() for _ in range(100)]
        for theta in (np.pi, np.pi - 1e-7, np.pi - 1e-3, -np.pi + 1e-5, 1e-6):
            transforms.append(
                kin.SE2(kin.exp_so2(theta), np.random.standard_normal(2))
            )
        return transforms

    def test_exp_matches_pinocchio(self):
        """exp_se2 matches integration on Pinocchio's SE2 Lie group."""
        se2 = pin.liegroups.SE2()
        for nu in self.planar_twists():
            with self.subTest(omega=nu[2]):
                self.assertTrue(
                    np.allclose(
                        to_config(kin.exp_se2(nu)),
                        se2.integrate(se2.neutral, nu),
                        atol=1e-12,
                    )
                )

    def test_log_matches_pinocchio(self):
        """log_se2 and Jlog_se2 match Pinocchio, including near pi."""
        se2 = pin.liegroups.SE2()
        for M in self.planar_transforms():
            q = to_config(M)
            self.assertTrue(
                np.allclose(
                    kin.log_se2(M),
                    se2.difference(se2.neutral, q),
                    atol=1e-10,
                )
            )
            self.assertTrue(
                np.allclose(
                    kin.Jlog_se2(M),
                    se2.dDifference(se2.neutral, q, pin.ARG1),
                    atol=1e-10,
                )
            )

    def test_consistent_with_se3(self):
        """SE(2) maps and adjoints are the planar restriction of SE(3) ones.

        Planar transforms are embedded in the xy-plane, where their twists
        are the (vx, vy, omega_z) coordinates of 6D twists.
        """
        for nu in self.planar_twists():
            with self.subTest(omega=nu[2]):
                twist = np.zeros(6)
                twist[PLANAR] = nu
                self.assertTrue(
                    to_se3(kin.exp_se2(nu)).is_approx(kin.exp6(twist))
                )
        for M in self.planar_transforms():
            M3 = to_se3(M)
            self.assertTrue(
                np.allclose(
                    kin.log_se2(M), kin.log6(M3).vector[PLANAR], atol=1e-6
                )
            )
            self.assertTrue(
                np.allclose(
                    kin.Jlog_se2(M),
                    kin.Jlog6(M3)[np.ix_(PLANAR, PLANAR)],
                    atol=1e-6,
                )
            )
            self.assertTrue(
                np.allclose(
                    M.action, M3.action[np.ix_(PLANAR, PLANAR)], atol=1e-12
                )
            )
            self.assertTrue(
                np.allclose(
                    M.action_inverse,
                    M3.action_inverse[np.ix_(PLANAR, PLANAR)],
                    atol=1e-12,
                )
            )

    def test_jlog_finite_difference(self):
        """Jlog_se2 matches a finite-difference approximation.

        As Jlog6, Jlog_se2 is the right Jacobian: for a perturbation delta of
        M in its local frame, log(M exp(delta)) = log(M) + Jlog(M) delta.
        """
        eps = 1e-7
        for _ in range(20):
            M = kin.SE2.Random()
            J = kin.Jlog_se2(M)
            J_fd = np.empty((3, 3))
            log_M = kin.log_se2(M)
            for k in range(3):
                delta = np.zeros(3)
                delta[k] = eps
                M_pert = M * kin.exp_se2(delta)
                J_fd[:, k] = (kin.log_se2(M_pert) - log_M) / eps
            self.assertTrue(np.allclose(J, J_fd, atol=1e-5))

    def test_se2_actions(self):
        """SE2 composition, inverse and adjoint are consistent."""
        for _ in range(20):
            A, B = kin.SE2.Random(), kin.SE2.Random()
            self.assertTrue(
                np.allclose(
                    (A * B).homogeneous,
                    A.homogeneous @ B.homogeneous,
                    atol=1e-12,
                )
            )
            self.assertTrue(
                np.allclose(
                    A.act_inv(B).homogeneous,
                    A.inverse().homogeneous @ B.homogeneous,
                    atol=1e-12,
                )
            )
            self.assertTrue(
                np.allclose(
                    (A.inverse() * A).homogeneous, np.eye(3), atol=1e-12
                )
            )
            point = np.random.standard_normal(2)
            self.assertTrue(
                np.allclose(
                    A.act(point),
                    A.rotation @ point + A.translation,
                    atol=1e-12,
                )
            )
            self.assertTrue(
                np.allclose(A.act_inv(A.act(point)), point, atol=1e-12)
            )
            # Ad_A nu twists transform like A exp(nu) A^-1
            nu = 0.1 * np.random.standard_normal(3)
            self.assertTrue(
                kin.exp_se2(A.action @ nu).is_approx(
                    A * kin.exp_se2(nu) * A.inverse(), prec=1e-10
                )
            )

    def test_interpolate(self):
        """SE2.Interpolate goes from A to B along the geodesic."""
        se2 = pin.liegroups.SE2()
        for _ in range(20):
            A, B = kin.SE2.Random(), kin.SE2.Random()
            self.assertTrue(kin.SE2.Interpolate(A, B, 0.0).is_approx(A))
            self.assertTrue(
                kin.SE2.Interpolate(A, B, 1.0).is_approx(B, prec=1e-10)
            )
            alpha = np.random.uniform(0.0, 1.0)
            self.assertTrue(
                np.allclose(
                    to_config(kin.SE2.Interpolate(A, B, alpha)),
                    se2.interpolate(to_config(A), to_config(B), alpha),
                    atol=1e-10,
                )
            )

    def test_random_is_a_rigid_transform(self):
        """SE2.Random draws its rotations from SO(2)."""
        for _ in range(20):
            M = kin.SE2.Random()
            self.assertTrue(
                np.allclose(M.rotation @ M.rotation.T, np.eye(2), atol=1e-12)
            )
            self.assertAlmostEqual(np.linalg.det(M.rotation), 1.0, places=12)

    def test_copy_and_conversions(self):
        """Copies own their memory and conversions are homogeneous."""
        M = kin.SE2.Random()
        N = M.copy()
        N.translation[0] += 1.0
        self.assertFalse(M.is_approx(N))
        self.assertTrue(np.array_equal(np.asarray(M), M.homogeneous))
        self.assertTrue(np.array_equal(M.toarray(), M.homogeneous))
        self.assertEqual(np.asarray(M, dtype=np.float32).dtype, np.float32)
        self.assertTrue(kin.SE2.Identity().is_approx(kin.SE2()))
        self.assertIn("SE2", repr(M))
