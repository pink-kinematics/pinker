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

from pinker import kinematics as kin

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
        "freeflyer": kin.JointModelFreeFlyer(),
        "planar": kin.JointModelPlanar(),
    }[root]
    pin_model = (
        pin.buildModelFromUrdf(urdf_path, root_pin)
        if root_pin is not None
        else pin.buildModelFromUrdf(urdf_path)
    )
    kin_model = kin.build_model_from_urdf(urdf_path, root_mq)
    return pin_model, kin_model


class TestVersusPinocchio(unittest.TestCase):
    """Compare models and kinematics against Pinocchio's."""

    def test_urdf_model(self):
        """Models built from URDF match joint for joint, frame for frame."""
        for urdf_path in URDF_PATHS:
            for root in ROOT_JOINTS:
                with self.subTest(urdf=os.path.basename(urdf_path), root=root):
                    pin_model, kin_model = build_pair(urdf_path, root)
                    assert_models_equal(pin_model, kin_model)

    def test_kinematics(self):
        """Kinematics match on random configurations."""
        for urdf_path in URDF_PATHS:
            for root in ROOT_JOINTS:
                with self.subTest(urdf=os.path.basename(urdf_path), root=root):
                    pin_model, kin_model = build_pair(urdf_path, root)
                    rng = np.random.default_rng(42)
                    for _ in range(10):
                        q = random_configuration(pin_model, rng)
                        assert_kinematics_equal(pin_model, kin_model, q)
                        assert_liegroup_equal(pin_model, kin_model, q, rng)

    def test_spherical_joint(self):
        """Programmatic model with a spherical joint matches Pinocchio."""
        pin_model = pin.Model()
        kin_model = kin.Model()
        placement_1 = pin.SE3.Random()
        j1p = pin_model.addJoint(
            0, pin.JointModelSpherical(), placement_1, "shoulder"
        )
        j1m = kin_model.add_joint(
            0,
            kin.JointModelSpherical(),
            kin.SE3(placement_1.rotation, placement_1.translation),
            "shoulder",
        )
        pin_model.addJointFrame(j1p)
        kin_model.add_joint_frame(j1m)
        inertia = pin.Inertia.Random()
        pin_model.appendBodyToJoint(j1p, inertia, pin.SE3.Identity())
        kin_model.append_body_to_joint(
            j1m, kin.Inertia(inertia.mass, inertia.lever, inertia.inertia)
        )
        pin_model.addBodyFrame("upper_arm", j1p, pin.SE3.Identity(), -1)
        kin_model.add_body_frame("upper_arm", j1m)
        placement_2 = pin.SE3.Random()
        j2p = pin_model.addJoint(j1p, pin.JointModelRY(), placement_2, "elbow")
        j2m = kin_model.add_joint(
            j1m,
            kin.JointModelRY(),
            kin.SE3(placement_2.rotation, placement_2.translation),
            "elbow",
        )
        pin_model.addJointFrame(j2p)
        kin_model.add_joint_frame(j2m)
        pin_model.appendBodyToJoint(j2p, inertia, pin.SE3.Identity())
        kin_model.append_body_to_joint(
            j2m, kin.Inertia(inertia.mass, inertia.lever, inertia.inertia)
        )
        pin_model.addBodyFrame("forearm", j2p, pin.SE3.Identity(), -1)
        kin_model.add_body_frame("forearm", j2m)

        rng = np.random.default_rng(7)
        for _ in range(10):
            q = random_configuration(pin_model, rng)
            assert_kinematics_equal(pin_model, kin_model, q)
            assert_liegroup_equal(pin_model, kin_model, q, rng)

    def test_neutral_and_limits(self):
        """Neutral configuration and limit vectors match."""
        for root in ROOT_JOINTS:
            with self.subTest(root=root):
                pin_model, kin_model = build_pair(URDF_PATHS[0], root)
                self.assertTrue(
                    np.allclose(pin.neutral(pin_model), kin.neutral(kin_model))
                )
                self.assertTrue(
                    np.allclose(
                        pin_model.velocityLimit, kin_model.velocity_limit
                    )
                )

    def test_frame_lookup_matches(self):
        """Frame lookups agree with Pinocchio's, except on misses.

        Pinocchio returns the sentinel index ``nframes`` when a frame is not
        found, where our getters raise, consistently with the other Model
        getters.
        """
        pin_model, kin_model = build_pair(URDF_PATHS[0], "fixed")
        for frame in kin_model.frames:
            self.assertTrue(pin_model.existFrame(frame.name))
            self.assertTrue(kin_model.exist_frame(frame.name))
            self.assertEqual(
                pin_model.getFrameId(frame.name),
                kin_model.get_frame_id(frame.name),
            )
        self.assertFalse(kin_model.exist_frame("no_such_frame"))
        self.assertEqual(  # just for the record
            pin_model.getFrameId("no_such_frame"), pin_model.nframes
        )
        with self.assertRaises(ValueError):
            kin_model.get_frame_id("no_such_frame")
