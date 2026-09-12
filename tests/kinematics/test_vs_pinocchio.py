# SPDX-License-Identifier: Apache-2.0

"""Cross-validate pinker.kinematics against Pinocchio.

These tests require the ``pinocchio`` package (installed in the pixi test
environment) and compare every quantity pinker.kinematics computes against the
reference implementation.
"""

import os
import unittest

import numpy as np
import pinocchio as pin
from compare import (
    assert_kinematics_equal,
    assert_liegroup_equal,
    assert_models_equal,
    random_configuration,
)

from pinker import kinematics as mq

HERE = os.path.dirname(__file__)
# Extra URDFs from a Pinocchio checkout next to the pinker workspace, when
# present (these tests only run on the bundled URDF otherwise).
PINOCCHIO_MODELS = os.path.join(HERE, "..", "..", "..", "pinocchio", "models")

URDF_PATHS = [
    os.path.join(HERE, "wheeled.urdf"),
]
for name in ("simple_humanoid.urdf", "baxter_simple.urdf"):
    path = os.path.join(PINOCCHIO_MODELS, name)
    if os.path.exists(path):
        URDF_PATHS.append(path)

ROOT_JOINTS = ["fixed", "freeflyer", "planar"]


def build_pair(urdf_path: str, root: str):
    """Build the same model with Pinocchio and pinker.kinematics."""
    root_pin = {
        "fixed": None,
        "freeflyer": pin.JointModelFreeFlyer(),
        "planar": pin.JointModelPlanar(),
    }[root]
    root_mq = {
        "fixed": None,
        "freeflyer": mq.JointModelFreeFlyer(),
        "planar": mq.JointModelPlanar(),
    }[root]
    mp = (
        pin.buildModelFromUrdf(urdf_path, root_pin)
        if root_pin is not None
        else pin.buildModelFromUrdf(urdf_path)
    )
    mm = mq.buildModelFromUrdf(urdf_path, root_mq)
    return mp, mm


class TestVersusPinocchio(unittest.TestCase):
    """Compare models and kinematics against Pinocchio's."""

    def test_urdf_model(self):
        """Models built from URDF match joint for joint, frame for frame."""
        for urdf_path in URDF_PATHS:
            for root in ROOT_JOINTS:
                with self.subTest(urdf=os.path.basename(urdf_path), root=root):
                    mp, mm = build_pair(urdf_path, root)
                    assert_models_equal(mp, mm)

    def test_kinematics(self):
        """Kinematics match on random configurations."""
        for urdf_path in URDF_PATHS:
            for root in ROOT_JOINTS:
                with self.subTest(urdf=os.path.basename(urdf_path), root=root):
                    mp, mm = build_pair(urdf_path, root)
                    rng = np.random.default_rng(42)
                    for _ in range(10):
                        q = random_configuration(mp, rng)
                        assert_kinematics_equal(mp, mm, q)
                        assert_liegroup_equal(mp, mm, q, rng)

    def test_spherical_joint(self):
        """Programmatic model with a spherical joint matches Pinocchio."""
        mp = pin.Model()
        mm = mq.Model()
        placement_1 = pin.SE3.Random()
        j1p = mp.addJoint(
            0, pin.JointModelSpherical(), placement_1, "shoulder"
        )
        j1m = mm.addJoint(
            0,
            mq.JointModelSpherical(),
            mq.SE3(placement_1.rotation, placement_1.translation),
            "shoulder",
        )
        mp.addJointFrame(j1p)
        mm.addJointFrame(j1m)
        inertia = pin.Inertia.Random()
        mp.appendBodyToJoint(j1p, inertia, pin.SE3.Identity())
        mm.appendBodyToJoint(
            j1m, mq.Inertia(inertia.mass, inertia.lever, inertia.inertia)
        )
        mp.addBodyFrame("upper_arm", j1p, pin.SE3.Identity(), -1)
        mm.addBodyFrame("upper_arm", j1m)
        placement_2 = pin.SE3.Random()
        j2p = mp.addJoint(j1p, pin.JointModelRY(), placement_2, "elbow")
        j2m = mm.addJoint(
            j1m,
            mq.JointModelRY(),
            mq.SE3(placement_2.rotation, placement_2.translation),
            "elbow",
        )
        mp.addJointFrame(j2p)
        mm.addJointFrame(j2m)
        mp.appendBodyToJoint(j2p, inertia, pin.SE3.Identity())
        mm.appendBodyToJoint(
            j2m, mq.Inertia(inertia.mass, inertia.lever, inertia.inertia)
        )
        mp.addBodyFrame("forearm", j2p, pin.SE3.Identity(), -1)
        mm.addBodyFrame("forearm", j2m)

        rng = np.random.default_rng(7)
        for _ in range(10):
            q = random_configuration(mp, rng)
            assert_kinematics_equal(mp, mm, q)
            assert_liegroup_equal(mp, mm, q, rng)

    def test_neutral_and_limits(self):
        """Neutral configuration and limit vectors match."""
        for root in ROOT_JOINTS:
            with self.subTest(root=root):
                mp, mm = build_pair(URDF_PATHS[0], root)
                self.assertTrue(np.allclose(pin.neutral(mp), mq.neutral(mm)))
                self.assertTrue(
                    np.allclose(mp.velocityLimit, mm.velocityLimit)
                )

    def test_frame_lookup_matches(self):
        """existFrame/getFrameId behave like Pinocchio, including misses."""
        mp, mm = build_pair(URDF_PATHS[0], "fixed")
        for frame in mm.frames:
            self.assertTrue(mp.existFrame(frame.name))
            self.assertTrue(mm.existFrame(frame.name))
            self.assertEqual(
                mp.getFrameId(frame.name), mm.getFrameId(frame.name)
            )
        self.assertFalse(mm.existFrame("no_such_frame"))
        self.assertEqual(mm.getFrameId("no_such_frame"), mm.nframes)
