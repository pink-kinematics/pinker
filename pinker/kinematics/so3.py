# SPDX-License-Identifier: Apache-2.0

"""Conversions between representations of rotations.

Quaternions are laid out as (x, y, z, w), vector part first, as in
configuration vectors. The one exception is :func:`quaternion_wxyz`,
named after the scalar-first order that Viser expects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # numpy.typing requires NumPy >= 1.20, we support >= 1.19
    from numpy.typing import ArrayLike


def quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) of a rotation matrix, by Shepperd's method.

    Args:
        R: Rotation matrix.

    Returns:
        Corresponding unit quaternion, as (w, x, y, z).
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        return np.array(
            [
                0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
            ]
        )
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array(
            [
                (R[2, 1] - R[1, 2]) / s,
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
            ]
        )
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array(
            [
                (R[0, 2] - R[2, 0]) / s,
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
            ]
        )
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array(
        [
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        ]
    )


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix from roll-pitch-yaw angles (extrinsic x-y-z).

    Args:
        roll: Rotation angle about the x-axis in [rad].
        pitch: Rotation angle about the y-axis in [rad].
        yaw: Rotation angle about the z-axis in [rad].

    Returns:
        Corresponding rotation matrix.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def quaternion_to_matrix(quat: ArrayLike) -> np.ndarray:
    """Rotation matrix of a unit quaternion.

    Args:
        quat: Unit quaternion, as (x, y, z, w) with its vector part first,
            the way quaternions are laid out in configuration vectors.

    Returns:
        Corresponding rotation matrix.
    """
    x, y, z, w = quat
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    )


def skew(v: ArrayLike) -> np.ndarray:
    """Skew-symmetric matrix of a 3D vector.

    Args:
        v: Three-dimensional vector.

    Returns:
        Matrix such that multiplying it by a vector w yields the cross
        product of v and w.
    """
    x, y, z = v
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )
