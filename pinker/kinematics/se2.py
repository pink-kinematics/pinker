# SPDX-License-Identifier: Apache-2.0

"""Planar rigid transforms.

Planar twists are laid out as (vx, vy, omega), linear part first, as in the
tangent space of planar joints.
"""

from typing import Optional, Union

import numpy as np

from . import _kinematics_c as _c


class SE2:
    """Planar rigid transform, by a 2D rotation matrix and a translation."""

    __slots__ = ("rotation", "translation")

    def __init__(
        self,
        rotation: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
    ):
        """Initialize transform.

        Args:
            rotation: 2x2 rotation matrix (default: identity).
            translation: translation vector (default: zero).
        """
        if rotation is None:
            rotation = np.eye(2)
        if translation is None:
            translation = np.zeros(2)
        self.rotation = np.asarray(rotation, dtype=np.float64).reshape(2, 2)
        self.translation = np.asarray(translation, dtype=np.float64).reshape(2)

    @classmethod
    def _new(cls, rotation: np.ndarray, translation: np.ndarray) -> "SE2":
        """Fast internal constructor for well-formed arrays.

        Args:
            rotation: 2x2 rotation matrix, used as is.
            translation: Translation vector of size 2, used as is.

        Returns:
            Transform sharing memory with the input arrays.
        """
        M = cls.__new__(cls)
        M.rotation = rotation
        M.translation = translation
        return M

    @classmethod
    def Identity(cls) -> "SE2":
        """Identity transform.

        Returns:
            Transform with identity rotation and zero translation.
        """
        return cls()

    @classmethod
    def Random(cls) -> "SE2":
        """Random transform, with translation in the unit square.

        Returns:
            Transform with a random rotation and a random translation.
        """
        theta = np.random.uniform(-np.pi, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, -s], [s, c]])
        return cls(rotation, np.random.uniform(-1.0, 1.0, 2))

    @classmethod
    def Interpolate(cls, A: "SE2", B: "SE2", alpha: float) -> "SE2":
        """Interpolate between two transforms.

        Args:
            A: First transform.
            B: Second transform.
            alpha: Interpolation parameter in [0, 1].

        Returns:
            Transform A * exp(alpha * log(A^-1 B)).
        """
        nu = _c.log_se2(*_se2_c_args(A.act_inv(B)))
        R, p = _c.exp_se2(np.ascontiguousarray(alpha * nu))
        return A * cls(R, p)

    def copy(self) -> "SE2":
        """Copy of this transform with its own memory.

        Returns:
            New transform with the same value in freshly allocated arrays.
        """
        return SE2._new(self.rotation.copy(), self.translation.copy())

    def inverse(self) -> "SE2":
        """Inverse transform.

        Returns:
            Transform undoing this one.
        """
        Rt = self.rotation.T.copy()
        return SE2._new(Rt, -Rt @ self.translation)

    def act(self, other: Union["SE2", np.ndarray]):
        """Compose with a transform, or apply to a point.

        Args:
            other: Transform to compose with, or point to apply this
                transform to.

        Returns:
            Composed transform, or transformed point.
        """
        if isinstance(other, SE2):
            return SE2._new(
                self.rotation @ other.rotation,
                self.translation + self.rotation @ other.translation,
            )
        return self.rotation @ other + self.translation

    def act_inv(self, other: Union["SE2", np.ndarray]):
        """Compose the inverse with a transform, or apply it to a point.

        Args:
            other: Transform to compose the inverse with, or point to apply
                the inverse transform to.

        Returns:
            Composed transform, or transformed point.
        """
        Rt = self.rotation.T
        if isinstance(other, SE2):
            return SE2._new(
                Rt @ other.rotation,
                Rt @ (other.translation - self.translation),
            )
        return Rt @ (other - self.translation)

    def __mul__(self, other: "SE2") -> "SE2":
        """Compose two transforms.

        Args:
            other: Transform to compose with.

        Returns:
            Composed transform.
        """
        return self.act(other)

    @property
    def homogeneous(self) -> np.ndarray:
        """Homogeneous 3x3 matrix of the transform.

        Returns:
            Homogeneous matrix of the transform.
        """
        H = np.eye(3)
        H[:2, :2] = self.rotation
        H[:2, 2] = self.translation
        return H

    def toarray(self) -> np.ndarray:
        """Convert to a NumPy array as the homogeneous 3x3 matrix.

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
        """Adjoint 3x3 matrix, mapping (vx, vy, omega) twists.

        Returns:
            Adjoint matrix of the transform.
        """
        A = np.zeros((3, 3))
        A[:2, :2] = self.rotation
        A[0, 2] = self.translation[1]
        A[1, 2] = -self.translation[0]
        A[2, 2] = 1.0
        return A

    @property
    def action_inverse(self) -> np.ndarray:
        """Adjoint 3x3 matrix of the inverse transform.

        Returns:
            Adjoint matrix of the inverse transform.
        """
        return self.inverse().action

    def is_approx(self, other: "SE2", prec: float = 1e-12) -> bool:
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
            f"SE2(rotation=\n{self.rotation},\ntranslation={self.translation})"
        )


def _se2_c_args(M: SE2):
    """Contiguous (rotation, translation) arrays for the C kernels.

    Args:
        M: Planar rigid transform.

    Returns:
        Pair of C-contiguous rotation and translation arrays.
    """
    return (
        np.ascontiguousarray(M.rotation),
        np.ascontiguousarray(M.translation),
    )
