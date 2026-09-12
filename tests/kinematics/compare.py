# SPDX-License-Identifier: Apache-2.0

"""Helpers comparing pinker.kinematics and Pinocchio models and algorithms."""

import numpy as np
import pinocchio as pin

from pinker import kinematics as kin

TOL = 1e-10

REFERENCE_FRAMES = (
    (pin.ReferenceFrame.WORLD, kin.ReferenceFrame.WORLD),
    (pin.ReferenceFrame.LOCAL, kin.ReferenceFrame.LOCAL),
    (
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        kin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    ),
)


def assert_models_equal(pin_model: pin.Model, kin_model: kin.Model) -> None:
    """Check that a pinker.kinematics model matches a Pinocchio model."""
    assert (
        pin_model.nq,
        pin_model.nv,
        pin_model.njoints,
        pin_model.nframes,
    ) == (
        kin_model.nq,
        kin_model.nv,
        kin_model.njoints,
        kin_model.nframes,
    )
    assert list(pin_model.names) == list(kin_model.names)
    assert list(pin_model.parents) == list(kin_model.parents)
    for jp, jm in zip(pin_model.joints, kin_model.joints):
        assert (jp.idx_q, jp.idx_v) == (jm.idx_q, jm.idx_v)
    # Skip the universe joint: Pinocchio's reports nq = nv = 1, the backend's
    # reports 0. Pink only reads joints with idx_q >= 0.
    for i in range(1, pin_model.njoints):
        assert (pin_model.joints[i].nq, pin_model.joints[i].nv) == (
            kin_model.joints[i].nq,
            kin_model.joints[i].nv,
        )
        assert (
            pin_model.joints[i].shortname() == kin_model.joints[i].shortname()
        )
        assert np.allclose(
            pin_model.jointPlacements[i].homogeneous,
            kin_model.jointPlacements[i].homogeneous,
            atol=TOL,
        )
    assert [f.name for f in pin_model.frames] == [
        f.name for f in kin_model.frames
    ]
    for fp, fm in zip(pin_model.frames, kin_model.frames):
        assert fp.parentJoint == fm.parentJoint
        assert int(fp.type) == int(fm.type)
        assert np.allclose(
            fp.placement.homogeneous, fm.placement.homogeneous, atol=TOL
        )
    assert np.allclose(
        pin_model.lowerPositionLimit, kin_model.lowerPositionLimit
    )
    assert np.allclose(
        pin_model.upperPositionLimit, kin_model.upperPositionLimit
    )
    assert np.allclose(pin_model.velocityLimit, kin_model.velocityLimit)
    assert list(pin_model.hasConfigurationLimit()) == list(
        kin_model.has_configuration_limit()
    )
    for ip, im in zip(pin_model.inertias, kin_model.inertias):
        assert np.isclose(ip.mass, im.mass)
        if ip.mass > 0.0:
            assert np.allclose(ip.lever, im.lever, atol=TOL)
    assert np.allclose(pin.neutral(pin_model), kin.neutral(kin_model))


def assert_kinematics_equal(
    pin_model: pin.Model, kin_model: kin.Model, q: np.ndarray
) -> None:
    """Check kinematics quantities on one configuration."""
    pin_data, kin_data = pin_model.createData(), kin_model.create_data()
    pin.computeJointJacobians(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    kin.compute_joint_jacobians(kin_model, kin_data, q)
    kin.update_frame_placements(kin_model, kin_data)
    for j in range(pin_model.njoints):
        assert np.allclose(
            pin_data.oMi[j].homogeneous, kin_data.oMi[j].homogeneous, atol=TOL
        )
    for f in range(pin_model.nframes):
        assert np.allclose(
            pin_data.oMf[f].homogeneous, kin_data.oMf[f].homogeneous, atol=TOL
        )
    assert np.allclose(pin_data.J, kin_data.J, atol=TOL)
    for f in range(pin_model.nframes):
        for rf_pin, rf_kin in REFERENCE_FRAMES:
            Jp = pin.getFrameJacobian(pin_model, pin_data, f, rf_pin)
            Jm = kin.get_frame_jacobian(kin_model, kin_data, f, rf_kin)
            assert np.allclose(Jp, Jm, atol=TOL), (f, rf_pin)
    for j in range(1, pin_model.njoints):
        for rf_pin, rf_kin in REFERENCE_FRAMES:
            Jp = pin.getJointJacobian(pin_model, pin_data, j, rf_pin)
            Jm = kin.get_joint_jacobian(kin_model, kin_data, j, rf_kin)
            assert np.allclose(Jp, Jm, atol=TOL), (j, rf_pin)
    assert np.allclose(
        pin.centerOfMass(pin_model, pin_data, q),
        kin.center_of_mass(kin_model, kin_data, q),
        atol=TOL,
    )
    assert np.allclose(
        pin.jacobianCenterOfMass(pin_model, pin_data, q),
        kin.jacobian_center_of_mass(kin_model, kin_data, q),
        atol=TOL,
    )


def assert_liegroup_equal(
    pin_model: pin.Model,
    kin_model: kin.Model,
    q: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Check configuration-space operations around one configuration."""
    v = rng.standard_normal(pin_model.nv)
    q_pin = pin.integrate(pin_model, q, v)
    q_kin = kin.integrate(kin_model, q, v)
    assert np.allclose(q_pin, q_kin, atol=TOL)
    q2 = q_pin
    assert np.allclose(
        pin.difference(pin_model, q, q2),
        kin.difference(kin_model, q, q2),
        atol=TOL,
    )
    assert np.allclose(
        pin.dDifference(pin_model, q, q2, pin.ARG0),
        kin.d_difference(kin_model, q, q2, kin.ARG0),
        atol=TOL,
    )
    assert np.allclose(
        pin.dDifference(pin_model, q, q2, pin.ARG1),
        kin.d_difference(kin_model, q, q2, kin.ARG1),
        atol=TOL,
    )


def random_configuration(
    pin_model: pin.Model, rng: np.random.Generator
) -> np.ndarray:
    """Random valid configuration, built by integrating a random tangent."""
    return pin.integrate(
        pin_model,
        pin.neutral(pin_model),
        2.0 * rng.standard_normal(pin_model.nv),
    )
