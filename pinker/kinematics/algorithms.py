# SPDX-License-Identifier: Apache-2.0

"""Kinematics algorithms.

Heavy computations are delegated to the C extension. Configuration vectors are
laid out per joint as described in :mod:`pinker.kinematics.joints`.

Functions take configuration and tangent vectors as array-likes, which they
copy into contiguous float64 arrays, and return NumPy arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from . import _kinematics_c as _c
from .model import Data, Model, ReferenceFrame
from .se3 import SE3, Motion, _se3_c_args

if TYPE_CHECKING:  # numpy.typing requires NumPy >= 1.20, we support >= 1.19
    from numpy.typing import ArrayLike

ARG0 = 0
ARG1 = 1


def _vec(x: ArrayLike, n: Optional[int] = None) -> np.ndarray:
    """Contiguous float64 copy-free view of a vector argument.

    Args:
        x: Array-like to view as a flat vector.
        n: Expected number of elements, if it should be checked.

    Returns:
        Flat C-contiguous float64 view of ``x``.

    Raises:
        ValueError: If ``n`` is given and ``x`` has a different size.
    """
    v = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    if n is not None and v.shape[0] != n:
        raise ValueError(f"expected a vector of size {n}, got {v.shape[0]}")
    return v


def neutral(model: Model) -> np.ndarray:
    """Neutral configuration of a model.

    Args:
        model: Robot model.

    Returns:
        Neutral configuration vector of the model.
    """
    from .joints import joint_neutral

    q = np.zeros(model.nq)
    for joint in model.joints[1:]:
        q[joint.idx_q : joint.idx_q + joint.nq] = joint_neutral(joint.jtype)
    return q


def custom_configuration(model: Model, **kwargs) -> np.ndarray:
    """Generate a configuration vector where named joints have given values.

    Args:
        model: Robot model.
        kwargs: Custom values for joint coordinates.

    Returns:
        Configuration vector where named joints have the values specified in
        keyword arguments, and other joints have their neutral value.

    Raises:
        ValueError: If a joint value does not have the dimension of the
            corresponding joint.
    """
    q = neutral(model)
    for name, value in kwargs.items():
        joint_id = model.get_joint_id(name)
        joint = model.joints[joint_id]
        value = np.array(value).flatten()
        if value.shape[0] != joint.nq:
            raise ValueError(
                f"Joint '{name}' has {joint.nq=} but is set to {value.shape=}"
            )
        q[joint.idx_q : joint.idx_q + joint.nq] = value
    return q


def forward_kinematics(model: Model, data: Data, q: ArrayLike) -> None:
    """Compute joint placements for a configuration.

    Args:
        model: Robot model.
        data: Data of the model, updated in place with joint placements.
        q: Configuration vector.
    """
    packed = model._packed()
    _c.forward_kinematics(
        packed["jtype"],
        packed["parent"],
        packed["idx_q"],
        packed["axis"],
        packed["jp_rot"],
        packed["jp_trans"],
        _vec(q, model.nq),
        data._oMi_rot,
        data._oMi_trans,
    )


def compute_joint_jacobians(
    model: Model, data: Data, q: Optional[ArrayLike] = None
) -> np.ndarray:
    """Compute joint placements and the full model Jacobian.

    As in Pinocchio, the full Jacobian ``data.J`` has one column per tangent
    coordinate, expressed in the world frame.

    Args:
        model: Robot model.
        data: Data of the model, updated in place.
        q: Configuration vector. If None, joint placements already in ``data``
            are used as is.

    Returns:
        Full model Jacobian ``data.J``, of shape (6, nv).
    """
    packed = model._packed()
    if q is not None:
        forward_kinematics(model, data, q)
    _c.joint_jacobians(
        packed["jtype"],
        packed["idx_v"],
        packed["axis"],
        data._oMi_rot,
        data._oMi_trans,
        data.J,
        model.nv,
    )
    return data.J


def update_frame_placements(model: Model, data: Data) -> None:
    """Update frame placements from joint placements.

    Args:
        model: Robot model.
        data: Data of the model, whose joint placements have already been
            computed, updated in place with frame placements.
    """
    packed = model._packed()
    _c.frame_placements(
        packed["fparent"],
        packed["fp_rot"],
        packed["fp_trans"],
        data._oMi_rot,
        data._oMi_trans,
        data._oMf_rot,
        data._oMf_trans,
    )


def frames_forward_kinematics(model: Model, data: Data, q: ArrayLike) -> None:
    """Compute joint and frame placements for a configuration.

    Args:
        model: Robot model.
        data: Data of the model, updated in place.
        q: Configuration vector.
    """
    forward_kinematics(model, data, q)
    update_frame_placements(model, data)


def get_frame_jacobian(
    model: Model,
    data: Data,
    frame_id: int,
    reference_frame: ReferenceFrame = ReferenceFrame.LOCAL,
) -> np.ndarray:
    """Jacobian of a frame, extracted from the full model Jacobian.

    Requires ``compute_joint_jacobians`` and ``update_frame_placements`` to
    have been called on the data beforehand.

    Args:
        model: Robot model.
        data: Data of the model, with up-to-date Jacobian and placements.
        frame_id: Index of the frame in ``model.frames``.
        reference_frame: Frame in which the Jacobian is expressed.

    Returns:
        Jacobian of the frame, of shape (6, nv).
    """
    packed = model._packed()
    frame = model.frames[frame_id]
    out = np.empty((6, model.nv))  # zeroed by the C kernel
    _c.frame_jacobian(
        data.J,
        data._oMf_rot[frame_id],
        data._oMf_trans[frame_id],
        packed["support_cols"][frame.parentJoint],
        int(reference_frame),
        out,
        model.nv,
    )
    return out


def get_joint_jacobian(
    model: Model,
    data: Data,
    joint_id: int,
    reference_frame: ReferenceFrame = ReferenceFrame.LOCAL,
) -> np.ndarray:
    """Jacobian of a joint, extracted from the full model Jacobian.

    Args:
        model: Robot model.
        data: Data of the model, with an up-to-date Jacobian.
        joint_id: Index of the joint in ``model.joints``.
        reference_frame: Frame in which the Jacobian is expressed.

    Returns:
        Jacobian of the joint, of shape (6, nv).
    """
    packed = model._packed()
    out = np.empty((6, model.nv))  # zeroed by the C kernel
    _c.frame_jacobian(
        data.J,
        data._oMi_rot[joint_id],
        data._oMi_trans[joint_id],
        packed["support_cols"][joint_id],
        int(reference_frame),
        out,
        model.nv,
    )
    return out


def integrate(model: Model, q: ArrayLike, v: ArrayLike) -> np.ndarray:
    """Integrate a tangent-space displacement from a configuration.

    Args:
        model: Robot model.
        q: Configuration vector to integrate from.
        v: Tangent-space displacement to integrate.

    Returns:
        Resulting configuration vector.
    """
    packed = model._packed()
    qout = np.empty(model.nq)
    _c.integrate(
        packed["jtype"],
        packed["idx_q"],
        packed["idx_v"],
        _vec(q, model.nq),
        _vec(v, model.nv),
        qout,
    )
    return qout


def difference(model: Model, q0: ArrayLike, q1: ArrayLike) -> np.ndarray:
    """Tangent-space difference going from q0 to q1.

    Args:
        model: Robot model.
        q0: Configuration to start from.
        q1: Configuration to go to.

    Returns:
        Tangent-space displacement ``v`` such that integrating ``v`` from
        ``q0`` yields ``q1``.
    """
    packed = model._packed()
    dout = np.empty(model.nv)
    _c.difference(
        packed["jtype"],
        packed["idx_q"],
        packed["idx_v"],
        _vec(q0, model.nq),
        _vec(q1, model.nq),
        dout,
    )
    return dout


def d_difference(
    model: Model, q0: ArrayLike, q1: ArrayLike, arg: int = ARG0
) -> np.ndarray:
    """Jacobian of the difference with respect to one of its arguments.

    Args:
        model: Robot model.
        q0: Configuration to start from.
        q1: Configuration to go to.
        arg: Differentiate with respect to ``q0`` (``ARG0``) or ``q1``
            (``ARG1``).

    Returns:
        Jacobian of the difference, of shape (nv, nv).
    """
    packed = model._packed()
    Jout = np.empty((model.nv, model.nv))
    _c.d_difference(
        packed["jtype"],
        packed["idx_q"],
        packed["idx_v"],
        _vec(q0, model.nq),
        _vec(q1, model.nq),
        int(arg),
        Jout,
        model.nv,
    )
    return Jout


def center_of_mass(
    model: Model, data: Data, q: Optional[ArrayLike] = None
) -> np.ndarray:
    """Center of mass of the robot, in the world frame.

    Args:
        model: Robot model.
        data: Data of the model, updated in place with the center of mass and
            the total mass.
        q: Configuration vector. If None, joint placements already in ``data``
            are used as is.

    Returns:
        Position of the center of mass in the world frame.
    """
    packed = model._packed()
    if q is not None:
        forward_kinematics(model, data, q)
    total_mass = _c.center_of_mass(
        packed["parent"],
        packed["mass"],
        packed["lever"],
        data._oMi_rot,
        data._oMi_trans,
        data.com[0],
    )
    data.mass[0] = total_mass
    return data.com[0]


def jacobian_center_of_mass(
    model: Model, data: Data, q: Optional[ArrayLike] = None
) -> np.ndarray:
    """Jacobian of the center of mass (3 x nv), in the world frame.

    Args:
        model: Robot model.
        data: Data of the model, updated in place with the center-of-mass
            Jacobian, the center of mass and the total mass.
        q: Configuration vector. If None, joint placements already in ``data``
            are used as is.

    Returns:
        Center-of-mass Jacobian ``data.Jcom``, of shape (3, nv).
    """
    packed = model._packed()
    if q is not None:
        forward_kinematics(model, data, q)
    compute_joint_jacobians(model, data)
    total_mass = _c.com_jacobian(
        packed["jtype"],
        packed["parent"],
        packed["idx_v"],
        packed["mass"],
        packed["lever"],
        data._oMi_rot,
        data._oMi_trans,
        data.J,
        data.Jcom,
        data.com[0],
        model.nv,
    )
    data.mass[0] = total_mass
    return data.Jcom


def log6(M: SE3) -> Motion:
    """Logarithm map of a rigid transform.

    Args:
        M: Rigid transform.

    Returns:
        Twist whose exponential is the transform.
    """
    return Motion(_c.log6(*_se3_c_args(M)))


def Jlog6(M: SE3) -> np.ndarray:
    """Jacobian of the SE3 logarithm at a rigid transform.

    Args:
        M: Rigid transform.

    Returns:
        Jacobian of :func:`log6` at the transform, of shape (6, 6).
    """
    return _c.Jlog6(*_se3_c_args(M))


def exp6(nu: Union[Motion, ArrayLike]) -> SE3:
    """Exponential map of a twist.

    Args:
        nu: Twist, as a :class:`.Motion` or a vector of size 6.

    Returns:
        Rigid transform whose logarithm is the twist.
    """
    vector = nu.vector if isinstance(nu, Motion) else nu
    R, p = _c.exp6(_vec(vector, 6))
    return SE3(R, p)


def log3(R: ArrayLike) -> np.ndarray:
    """Logarithm map of a rotation matrix.

    Args:
        R: Rotation matrix.

    Returns:
        Rotation vector, whose norm is the rotation angle.
    """
    return _c.log3(np.ascontiguousarray(R, dtype=np.float64))


def exp3(w: ArrayLike) -> np.ndarray:
    """Exponential map of a rotation vector.

    Args:
        w: Rotation vector, whose norm is the rotation angle.

    Returns:
        Corresponding rotation matrix.
    """
    return _c.exp3(_vec(w, 3))
