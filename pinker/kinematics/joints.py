# SPDX-License-Identifier: Apache-2.0

"""Joint models.

Joint type identifiers match the enumeration in the C extension. Quantities per
joint type (configuration and tangent dimensions, neutral configuration) follow
Pinocchio's conventions:

- Revolute and prismatic joints have one configuration variable.
- Unbounded (continuous) revolute joints store (cos, sin) of their angle.
- Spherical joints store a unit quaternion (x, y, z, w).
- Planar joints store (x, y, cos, sin).
- Free-flyer joints store (x, y, z, qx, qy, qz, qw).

Note that quaternions are stored with their vector part coming first.
"""

from typing import Optional

import numpy as np

JT_UNIVERSE = 0
JT_REVOLUTE = 1
JT_PRISMATIC = 2
JT_REVOLUTE_UNBOUNDED = 3
JT_SPHERICAL = 4
JT_PLANAR = 5
JT_FREEFLYER = 6

JOINT_NQ = {
    JT_UNIVERSE: 0,
    JT_REVOLUTE: 1,
    JT_PRISMATIC: 1,
    JT_REVOLUTE_UNBOUNDED: 2,
    JT_SPHERICAL: 4,
    JT_PLANAR: 4,
    JT_FREEFLYER: 7,
}

JOINT_NV = {
    JT_UNIVERSE: 0,
    JT_REVOLUTE: 1,
    JT_PRISMATIC: 1,
    JT_REVOLUTE_UNBOUNDED: 1,
    JT_SPHERICAL: 3,
    JT_PLANAR: 3,
    JT_FREEFLYER: 6,
}


def joint_neutral(jtype: int) -> np.ndarray:
    """Neutral configuration of a joint type.

    Args:
        jtype: Joint type identifier.

    Returns:
        Neutral configuration segment of the joint, of size ``nq``.
    """
    if jtype in (JT_REVOLUTE, JT_PRISMATIC):
        return np.zeros(1)
    if jtype == JT_REVOLUTE_UNBOUNDED:
        return np.array([1.0, 0.0])
    if jtype == JT_SPHERICAL:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if jtype == JT_PLANAR:
        return np.array([0.0, 0.0, 1.0, 0.0])
    if jtype == JT_FREEFLYER:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return np.zeros(0)


def joint_has_configuration_limit(jtype: int) -> list:
    """Which configuration variables of a joint have position limits.

    Args:
        jtype: Joint type identifier.

    Returns:
        One boolean per configuration variable of the joint, True if that
        variable has a position limit.
    """
    if jtype in (JT_REVOLUTE, JT_PRISMATIC):
        return [True]
    if jtype == JT_REVOLUTE_UNBOUNDED:
        return [False, False]
    if jtype == JT_SPHERICAL:
        return [False] * 4
    if jtype == JT_PLANAR:
        return [True, True, False, False]
    if jtype == JT_FREEFLYER:
        return [True] * 3 + [False] * 4
    return []


def _shortname(jtype: int, axis: Optional[np.ndarray]) -> str:
    """Pinocchio-compatible short name of a joint model.

    Args:
        jtype: Joint type identifier.
        axis: Joint axis, for revolute and prismatic joints.

    Returns:
        Short name of the joint model, e.g. "JointModelRZ" for a revolute
        joint about the z-axis.
    """
    if jtype == JT_SPHERICAL:
        return "JointModelSpherical"
    if jtype == JT_PLANAR:
        return "JointModelPlanar"
    if jtype == JT_FREEFLYER:
        return "JointModelFreeFlyer"
    if jtype == JT_UNIVERSE:
        return "JointModelUniverse"
    suffix = "Unaligned"
    if axis is not None:
        for name, unit in (
            ("X", (1, 0, 0)),
            ("Y", (0, 1, 0)),
            ("Z", (0, 0, 1)),
        ):
            if np.allclose(axis, unit):
                suffix = name
    if jtype == JT_REVOLUTE:
        prefix = "JointModelR"
        if suffix == "Unaligned":
            return "JointModelRevoluteUnaligned"
    elif jtype == JT_PRISMATIC:
        prefix = "JointModelP"
        if suffix == "Unaligned":
            return "JointModelPrismaticUnaligned"
    else:  # JT_REVOLUTE_UNBOUNDED
        prefix = "JointModelRUB"
        if suffix == "Unaligned":
            return "JointModelRevoluteUnboundedUnaligned"
    return prefix + suffix


class JointModel:
    """Description of a joint, before or after it is added to a model.

    Index attributes are set when the joint is added to a model, and are -1
    until then.

    Attributes:
        axis: Joint axis, for revolute and prismatic joints, None otherwise.
        id: Index of the joint in the model it belongs to.
        idx_q: Index of the joint in the configuration vector, so that its
            coordinates are ``q[idx_q:idx_q + nq]``.
        idx_v: Index of the joint in the tangent vector, so that its
            coordinates are ``v[idx_v:idx_v + nv]``.
        jtype: Joint type identifier.
    """

    axis: Optional[np.ndarray]
    id: int
    idx_q: int
    idx_v: int
    jtype: int

    def __init__(self, jtype: int, axis: Optional[np.ndarray] = None):
        """Initialize joint model from its type and optional axis.

        Args:
            jtype: Joint type identifier.
            axis: Joint axis, for revolute and prismatic joints. It is
                normalized upon construction.
        """
        if axis is not None:
            axis = np.asarray(axis, dtype=np.float64).reshape(3)
            axis = axis / np.linalg.norm(axis)
        self.jtype = jtype
        self.axis = axis
        self.id = -1
        self.idx_q = -1
        self.idx_v = -1

    @property
    def nq(self) -> int:
        """Number of configuration variables.

        Returns:
            Dimension of the configuration segment of the joint.
        """
        return JOINT_NQ[self.jtype]

    @property
    def nv(self) -> int:
        """Number of tangent (velocity) variables.

        Returns:
            Dimension of the tangent segment of the joint.
        """
        return JOINT_NV[self.jtype]

    def shortname(self) -> str:
        """Pinocchio-compatible short name, e.g. "JointModelRZ".

        Returns:
            Short name of the joint model.
        """
        return _shortname(self.jtype, self.axis)

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Human-readable representation of the joint model.
        """
        return f"{self.shortname()}(idx_q={self.idx_q}, idx_v={self.idx_v})"


def JointModelRevoluteUnaligned(x=1.0, y=0.0, z=0.0) -> JointModel:
    """Revolute joint about an arbitrary axis.

    Args:
        x: First coordinate of the joint axis.
        y: Second coordinate of the joint axis.
        z: Third coordinate of the joint axis.

    Returns:
        Corresponding joint model.
    """
    return JointModel(JT_REVOLUTE, np.array([x, y, z]))


def JointModelPrismaticUnaligned(x=1.0, y=0.0, z=0.0) -> JointModel:
    """Prismatic joint along an arbitrary axis.

    Args:
        x: First coordinate of the joint axis.
        y: Second coordinate of the joint axis.
        z: Third coordinate of the joint axis.

    Returns:
        Corresponding joint model.
    """
    return JointModel(JT_PRISMATIC, np.array([x, y, z]))


def JointModelRX() -> JointModel:
    """Revolute joint about the x-axis.

    Returns:
        Corresponding joint model.
    """
    return JointModel(JT_REVOLUTE, np.array([1.0, 0.0, 0.0]))


def JointModelRY() -> JointModel:
    """Revolute joint about the y-axis.

    Returns:
        Corresponding joint model.
    """
    return JointModel(JT_REVOLUTE, np.array([0.0, 1.0, 0.0]))


def JointModelRZ() -> JointModel:
    """Revolute joint about the z-axis.

    Returns:
        Corresponding joint model.
    """
    return JointModel(JT_REVOLUTE, np.array([0.0, 0.0, 1.0]))


def JointModelPX() -> JointModel:
    """Prismatic joint along the x-axis.

    Returns:
        Corresponding joint model.
    """
    return JointModel(JT_PRISMATIC, np.array([1.0, 0.0, 0.0]))


def JointModelPY() -> JointModel:
    """Prismatic joint along the y-axis.

    Returns:
        Corresponding joint model.
    """
    return JointModel(JT_PRISMATIC, np.array([0.0, 1.0, 0.0]))


def JointModelPZ() -> JointModel:
    """Prismatic joint along the z-axis.

    Returns:
        Corresponding joint model.
    """
    return JointModel(JT_PRISMATIC, np.array([0.0, 0.0, 1.0]))


def JointModelSpherical() -> JointModel:
    """Ball joint, parameterized by a unit quaternion.

    Its configuration segment is the unit quaternion (x, y, z, w), vector part
    first, so that the neutral configuration of the joint is (0, 0, 0, 1).

    Returns:
        Corresponding joint model.
    """
    return JointModel(JT_SPHERICAL)


def JointModelPlanar() -> JointModel:
    """Planar joint: translation in the xy-plane, rotation about z.

    Returns:
        Corresponding joint model.
    """
    return JointModel(JT_PLANAR)


def JointModelFreeFlyer() -> JointModel:
    """Free-flyer joint: full 6-DoF rigid motion.

    Its configuration segment is (x, y, z, qx, qy, qz, qw): the position of
    the joint followed by its unit quaternion, vector part first, so that the
    neutral configuration of the joint is (0, 0, 0, 0, 0, 0, 1).

    Returns:
        Corresponding joint model.
    """
    return JointModel(JT_FREEFLYER)
