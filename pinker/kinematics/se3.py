# SPDX-License-Identifier: Apache-2.0

"""Rigid transforms, spatial vectors and inertias."""

from typing import Optional, Union

import numpy as np

from . import _kinematics_c as _c
from .so3 import quaternion_to_matrix, skew


class SE3:
    """Rigid transform, represented by a rotation matrix and a translation."""

    __slots__ = ("rotation", "translation")

    def __init__(
        self,
        rotation: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
    ):
        """Initialize transform.

        Args:
            rotation: 3x3 rotation matrix (default: identity).
            translation: translation vector (default: zero).
        """
        if rotation is None:
            rotation = np.eye(3)
        if translation is None:
            translation = np.zeros(3)
        self.rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(translation, dtype=np.float64).reshape(3)

    @classmethod
    def _new(cls, rotation: np.ndarray, translation: np.ndarray) -> "SE3":
        """Fast internal constructor for well-formed arrays.

        Args:
            rotation: 3x3 rotation matrix, used as is.
            translation: Translation vector of size 3, used as is.

        Returns:
            Transform sharing memory with the input arrays.
        """
        M = cls.__new__(cls)
        M.rotation = rotation
        M.translation = translation
        return M

    @classmethod
    def Identity(cls) -> "SE3":
        """Identity transform.

        Returns:
            Transform with identity rotation and zero translation.
        """
        return cls()

    @classmethod
    def Random(cls) -> "SE3":
        """Random transform, with translation in the unit cube.

        Returns:
            Transform with a random rotation and a random translation.
        """
        quat = np.random.randn(4)
        quat /= np.linalg.norm(quat)
        rotation = quaternion_to_matrix(quat)
        return cls(rotation, np.random.uniform(-1.0, 1.0, 3))

    @classmethod
    def Interpolate(cls, A: "SE3", B: "SE3", alpha: float) -> "SE3":
        """Interpolate between two transforms.

        Args:
            A: First transform.
            B: Second transform.
            alpha: Interpolation parameter in [0, 1].

        Returns:
            Transform A * exp(alpha * log(A^-1 B)).
        """
        nu = _c.log6(*_se3_c_args(A.act_inv(B)))
        R, p = _c.exp6(np.ascontiguousarray(alpha * nu))
        return A * cls(R, p)

    def copy(self) -> "SE3":
        """Copy of this transform with its own memory.

        Returns:
            New transform with the same value in freshly allocated arrays.
        """
        return SE3._new(self.rotation.copy(), self.translation.copy())

    def inverse(self) -> "SE3":
        """Inverse transform.

        Returns:
            Transform undoing this one.
        """
        Rt = self.rotation.T.copy()
        return SE3._new(Rt, -Rt @ self.translation)

    def act(self, other: Union["SE3", np.ndarray]):
        """Compose with a transform, or apply to a point.

        Args:
            other: Transform to compose with, or point to apply this
                transform to.

        Returns:
            Composed transform, or transformed point.
        """
        if isinstance(other, SE3):
            return SE3._new(
                self.rotation @ other.rotation,
                self.translation + self.rotation @ other.translation,
            )
        return self.rotation @ other + self.translation

    def act_inv(self, other: Union["SE3", np.ndarray]):
        """Compose the inverse with a transform, or apply it to a point.

        Args:
            other: Transform to compose the inverse with, or point to apply
                the inverse transform to.

        Returns:
            Composed transform, or transformed point.
        """
        Rt = self.rotation.T
        if isinstance(other, SE3):
            return SE3._new(
                Rt @ other.rotation,
                Rt @ (other.translation - self.translation),
            )
        return Rt @ (other - self.translation)

    def __mul__(self, other: "SE3") -> "SE3":
        """Compose two transforms.

        Args:
            other: Transform to compose with.

        Returns:
            Composed transform.
        """
        return self.act(other)

    @property
    def homogeneous(self) -> np.ndarray:
        """Homogeneous 4x4 matrix of the transform.

        Returns:
            Homogeneous matrix of the transform.
        """
        H = np.eye(4)
        H[:3, :3] = self.rotation
        H[:3, 3] = self.translation
        return H

    def toarray(self) -> np.ndarray:
        """Convert to a NumPy array as the homogeneous 4x4 matrix.

        The transform stores its rotation and translation separately, so the
        array returned by this function is a copy and modifying it won't affect
        the transform.

        Returns:
            Homogeneous matrix of the transform.
        """
        return self.homogeneous

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """NumPy conversion as the homogeneous matrix.

        Args:
            dtype: Data type of the output array, if it should be cast.
            copy: Unused, for compatibility with the NumPy 2 protocol.

        Returns:
            Homogeneous matrix of the transform.
        """
        H = self.homogeneous
        return H if dtype is None else H.astype(dtype)

    @property
    def action(self) -> np.ndarray:
        """Adjoint 6x6 matrix, mapping (linear, angular) twists.

        Returns:
            Adjoint matrix of the transform.
        """
        R, p = self.rotation, self.translation
        A = np.zeros((6, 6))
        A[:3, :3] = R
        A[3:, 3:] = R
        A[:3, 3:] = skew(p) @ R
        return A

    @property
    def action_inverse(self) -> np.ndarray:
        """Adjoint 6x6 matrix of the inverse transform.

        Returns:
            Adjoint matrix of the inverse transform.
        """
        return self.inverse().action

    def is_approx(self, other: "SE3", prec: float = 1e-12) -> bool:
        """Check if two transforms are approximately equal.

        Args:
            other: Transform to compare with.
            prec: Absolute tolerance on rotation and translation coefficients.

        Returns:
            True if the two transforms are equal up to the tolerance.
        """
        return np.allclose(
            self.rotation, other.rotation, atol=prec
        ) and np.allclose(self.translation, other.translation, atol=prec)

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Human-readable representation of the transform.
        """
        return (
            f"SE3(rotation=\n{self.rotation},\ntranslation={self.translation})"
        )


def _se3_c_args(M: SE3):
    """Contiguous (rotation, translation) arrays for the C kernels.

    Args:
        M: Rigid transform.

    Returns:
        Pair of C-contiguous rotation and translation arrays.
    """
    return (
        np.ascontiguousarray(M.rotation),
        np.ascontiguousarray(M.translation),
    )


class Motion:
    """Spatial velocity (linear, angular)."""

    __slots__ = ("vector",)

    def __init__(self, vector: np.ndarray):
        """Initialize from a 6D vector (linear, angular).

        Args:
            vector: Spatial velocity, as (linear[3], angular[3]).
        """
        self.vector = np.asarray(vector, dtype=np.float64).reshape(6)

    @property
    def linear(self) -> np.ndarray:
        """Linear part.

        Returns:
            Linear part of the spatial velocity.
        """
        return self.vector[:3]

    @property
    def angular(self) -> np.ndarray:
        """Angular part.

        Returns:
            Angular part of the spatial velocity.
        """
        return self.vector[3:]

    def asarray(self) -> np.ndarray:
        """Convert to a NumPy array as a 6-dimensional vector.

        Note that this is not a copy: the returned array is directly the stored
        representation of the motion vector, and writing to it updates the
        motion directly. Copy it (or call :func:`numpy.array`) if that is not
        what you want.

        Returns:
            Spatial velocity as a vector of size 6.
        """
        return self.vector

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """NumPy conversion as the spatial velocity vector.

        Args:
            dtype: Data type of the output array, if it should be cast.
            copy: Unused, for compatibility with the NumPy 2 protocol.

        Returns:
            Spatial velocity as a vector of size 6.
        """
        v = self.vector
        return v if dtype is None else v.astype(dtype)

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Human-readable representation of the spatial velocity.
        """
        return f"Motion({self.vector})"


class Inertia:
    """Spatial inertia: mass, center of mass and rotational inertia."""

    __slots__ = ("mass", "lever", "inertia")

    def __init__(
        self,
        mass: float = 0.0,
        lever: Optional[np.ndarray] = None,
        inertia: Optional[np.ndarray] = None,
    ):
        """Initialize inertia.

        Args:
            mass: Mass in [kg].
            lever: Center of mass in the body frame.
            inertia: 3x3 rotational inertia at the center of mass.
        """
        self.mass = float(mass)
        self.lever = (
            np.zeros(3)
            if lever is None
            else np.asarray(lever, dtype=np.float64).reshape(3)
        )
        self.inertia = (
            np.zeros((3, 3))
            if inertia is None
            else np.asarray(inertia, dtype=np.float64).reshape(3, 3)
        )

    @classmethod
    def Zero(cls) -> "Inertia":
        """Zero inertia.

        Returns:
            Inertia with zero mass, lever and rotational inertia.
        """
        return cls()

    @classmethod
    def FromBox(cls, mass: float, x: float, y: float, z: float) -> "Inertia":
        """Inertia of a box of a given mass and dimensions.

        Args:
            mass: Mass of the box in [kg].
            x: Length of the box along the x-axis in [m].
            y: Length of the box along the y-axis in [m].
            z: Length of the box along the z-axis in [m].

        Returns:
            Inertia of the box at its center.
        """
        I_diag = (
            mass
            / 12.0
            * np.array([y * y + z * z, x * x + z * z, x * x + y * y])
        )
        return cls(mass, np.zeros(3), np.diag(I_diag))

    def displaced(self, placement: SE3) -> "Inertia":
        """This inertia expressed in a new frame.

        Args:
            placement: Transform from the body frame to the new frame.

        Returns:
            Inertia in the new frame.
        """
        R = placement.rotation
        lever = placement.act(self.lever)
        inertia = R @ self.inertia @ R.T
        return Inertia(self.mass, lever, inertia)

    def __add__(self, other: "Inertia") -> "Inertia":
        """Sum of two inertias expressed in the same frame.

        Args:
            other: Inertia to add to this one.

        Returns:
            Total inertia of the two bodies.
        """
        mass = self.mass + other.mass
        if mass <= 0.0:
            return Inertia()
        lever = (self.mass * self.lever + other.mass * other.lever) / mass

        def at_point(body: "Inertia", point: np.ndarray) -> np.ndarray:
            """Rotational inertia of a body about another point.

            Args:
                body: Body whose inertia is transported.
                point: Point to transport the inertia to.

            Returns:
                Rotational inertia of the body about the point.
            """
            r = body.lever - point
            return body.inertia + body.mass * (
                np.dot(r, r) * np.eye(3) - np.outer(r, r)
            )

        inertia = at_point(self, lever) + at_point(other, lever)
        return Inertia(mass, lever, inertia)
