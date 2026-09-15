# SPDX-License-Identifier: Apache-2.0

"""Exceptions specific to Pinker."""

import qpsolvers


class PinkerError(Exception):
    """Base class for Pinker exceptions."""


class FrameNotFound(PinkerError):
    """Exception raised when a frame is not found in the robot model."""

    def __init__(self, name: str, frames: list) -> None:
        """Create exception.

        Args:
            name: Name of the frame.
            frames: List of all robot frames.
        """
        self.name = name
        frame_names = [frame.name for frame in frames]
        self.message = (
            f'Name "{name}" is not a robot frame name in {frame_names}'
        )
        super().__init__(self.message)


class InvalidCollisionPairs(PinkerError):
    """IF the number of collision pairs is invalid."""


class NegativeMinimumDistance(PinkerError):
    """If the minimum distance in body spherical barrier is negative."""


class NoPositionLimitProvided(PinkerError):
    """If neither minimum nor maximum position limits are provided."""


class NoSolutionFound(PinkerError):
    """The QP solver did not find a solution to the differential IK problem."""

    def __init__(
        self,
        problem: qpsolvers.Problem,
        results: qpsolvers.Solution,
    ) -> None:
        """Create exception.

        Args:
            problem: Quadratic-programming problem, for investigation.
            results: Results returned by the backend QP solver.
        """
        super().__init__(
            "QP solver did not find a solution to the differential IK problem"
        )
        self.problem = problem
        self.results = results


class NotWithinConfigurationLimits(PinkerError):
    """Exception thrown when a robot configuration violates its limits.

    Attributes:
        joint: Index of the joint in the configuration vector.
        value: Invalid value of the joint.
        lower: Minimum allowed value for this joint.
        upper: Maximum allowed value for this joint.
    """

    joint: int
    value: float
    lower: float
    upper: float

    def __init__(
        self,
        joint: int,
        value: float,
        lower: float,
        upper: float,
    ) -> None:
        """Create exception.

        Args:
            joint: Index of the joint in the configuration vector.
            value: Invalid value of the joint.
            lower: Minimum allowed value for this joint.
            upper: Maximum allowed value for this joint.
        """
        self.joint = joint
        self.value = value
        self.lower = lower
        self.upper = upper
        self.message = (
            f"Joint {joint} violates configuration limits "
            f"{lower} <= {value} <= {upper}"
        )
        super().__init__(self.message)


class TargetNotSet(PinkerError):
    """Exception raised when attempting to compute with an unset target."""


class TaskDefinitionError(PinkerError):
    """Exception raised when a task definition is ill-formed."""


class TaskJacobianNotSet(PinkerError):
    """Exception raised when attempting to compute without a task Jacobian."""
