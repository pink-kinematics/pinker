#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers comparing pinker.kinematics and Pinocchio models and algorithms."""

import numpy as np
import pinocchio as pin

from pinker import kinematics as mq

TOL = 1e-10

REFERENCE_FRAMES = (
    (pin.ReferenceFrame.WORLD, mq.ReferenceFrame.WORLD),
    (pin.ReferenceFrame.LOCAL, mq.ReferenceFrame.LOCAL),
    (
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        mq.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    ),
)


def assert_models_equal(mp: pin.Model, mm: mq.Model) -> None:
    """Check that a pinker.kinematics model matches a Pinocchio model."""
    assert (mp.nq, mp.nv, mp.njoints, mp.nframes) == (
        mm.nq,
        mm.nv,
        mm.njoints,
        mm.nframes,
    )
    assert list(mp.names) == list(mm.names)
    assert list(mp.parents) == list(mm.parents)
    for jp, jm in zip(mp.joints, mm.joints):
        assert (jp.idx_q, jp.idx_v) == (jm.idx_q, jm.idx_v)
    # Skip the universe joint: Pinocchio's reports nq = nv = 1, the backend's
    # reports 0. Pink only reads joints with idx_q >= 0.
    for i in range(1, mp.njoints):
        assert (mp.joints[i].nq, mp.joints[i].nv) == (
            mm.joints[i].nq,
            mm.joints[i].nv,
        )
        assert mp.joints[i].shortname() == mm.joints[i].shortname()
        assert np.allclose(
            mp.jointPlacements[i].homogeneous,
            mm.jointPlacements[i].homogeneous,
            atol=TOL,
        )
    assert [f.name for f in mp.frames] == [f.name for f in mm.frames]
    for fp, fm in zip(mp.frames, mm.frames):
        assert fp.parentJoint == fm.parentJoint
        assert int(fp.type) == int(fm.type)
        assert np.allclose(
            fp.placement.homogeneous, fm.placement.homogeneous, atol=TOL
        )
    assert np.allclose(mp.lowerPositionLimit, mm.lowerPositionLimit)
    assert np.allclose(mp.upperPositionLimit, mm.upperPositionLimit)
    assert np.allclose(mp.velocityLimit, mm.velocityLimit)
    assert list(mp.hasConfigurationLimit()) == list(
        mm.hasConfigurationLimit()
    )
    for ip, im in zip(mp.inertias, mm.inertias):
        assert np.isclose(ip.mass, im.mass)
        if ip.mass > 0.0:
            assert np.allclose(ip.lever, im.lever, atol=TOL)
    assert np.allclose(pin.neutral(mp), mq.neutral(mm))


def assert_kinematics_equal(
    mp: pin.Model, mm: mq.Model, q: np.ndarray
) -> None:
    """Check kinematics quantities on one configuration."""
    dp, dm = mp.createData(), mm.createData()
    pin.computeJointJacobians(mp, dp, q)
    pin.updateFramePlacements(mp, dp)
    mq.computeJointJacobians(mm, dm, q)
    mq.updateFramePlacements(mm, dm)
    for j in range(mp.njoints):
        assert np.allclose(
            dp.oMi[j].homogeneous, dm.oMi[j].homogeneous, atol=TOL
        )
    for f in range(mp.nframes):
        assert np.allclose(
            dp.oMf[f].homogeneous, dm.oMf[f].homogeneous, atol=TOL
        )
    assert np.allclose(dp.J, dm.J, atol=TOL)
    for f in range(mp.nframes):
        for rf_pin, rf_mq in REFERENCE_FRAMES:
            Jp = pin.getFrameJacobian(mp, dp, f, rf_pin)
            Jm = mq.getFrameJacobian(mm, dm, f, rf_mq)
            assert np.allclose(Jp, Jm, atol=TOL), (f, rf_pin)
    for j in range(1, mp.njoints):
        for rf_pin, rf_mq in REFERENCE_FRAMES:
            Jp = pin.getJointJacobian(mp, dp, j, rf_pin)
            Jm = mq.getJointJacobian(mm, dm, j, rf_mq)
            assert np.allclose(Jp, Jm, atol=TOL), (j, rf_pin)
    assert np.allclose(
        pin.centerOfMass(mp, dp, q), mq.centerOfMass(mm, dm, q), atol=TOL
    )
    assert np.allclose(
        pin.jacobianCenterOfMass(mp, dp, q),
        mq.jacobianCenterOfMass(mm, dm, q),
        atol=TOL,
    )


def assert_liegroup_equal(
    mp: pin.Model, mm: mq.Model, q: np.ndarray, rng: np.random.Generator
) -> None:
    """Check configuration-space operations around one configuration."""
    v = rng.standard_normal(mp.nv)
    q_pin = pin.integrate(mp, q, v)
    q_mq = mq.integrate(mm, q, v)
    assert np.allclose(q_pin, q_mq, atol=TOL)
    q2 = q_pin
    assert np.allclose(
        pin.difference(mp, q, q2), mq.difference(mm, q, q2), atol=TOL
    )
    assert np.allclose(
        pin.dDifference(mp, q, q2, pin.ARG0),
        mq.dDifference(mm, q, q2, mq.ARG0),
        atol=TOL,
    )
    assert np.allclose(
        pin.dDifference(mp, q, q2, pin.ARG1),
        mq.dDifference(mm, q, q2, mq.ARG1),
        atol=TOL,
    )


def random_configuration(
    mp: pin.Model, rng: np.random.Generator
) -> np.ndarray:
    """Random valid configuration, built by integrating a random tangent."""
    return pin.integrate(mp, pin.neutral(mp), 2.0 * rng.standard_normal(mp.nv))
