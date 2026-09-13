# SPDX-License-Identifier: Apache-2.0

"""Configuration of a robot model.

Pinker uses :mod:`pinker.kinematics` for forward kinematics. A
:class:`Configuration` holds a robot model and data for this model where
forward kinematics have been run. This means that the geometric state of
the model has been computed, and quantities such as frame transforms and
frame Jacobians are available.
"""

import logging
from typing import List, Optional

import numpy as np

from . import kinematics as kin
from .exceptions import FrameNotFound, NotWithinConfigurationLimits
from .limits import ConfigurationLimit, Limit, VelocityLimit
from .utils import VectorSpace


class Configuration:
    """Type indicating that configuration-dependent quantities are available.

    In Pinker, this type enables access to frame transforms and frame
    Jacobians. We rely on typing to make sure the proper forward kinematics
    functions have been called beforehand:

    .. code:: python

        kin.compute_joint_jacobians(model, data, configuration)
        kin.update_frame_placements(model, data)

    The former computes the full model Jacobian into ``data.J``. (It also
    computes forward kinematics, so there is no need to further call
    ``kin.forward_kinematics(model, data, configuration)``.) The latter updates
    frame placements.

    Attributes:
        data: Data corresponding to :data:`Configuration.model`.
        model: Kinodynamic model.
        q: Configuration vector for the robot model.
    """

    data: kin.Data
    model: kin.Model
    q: np.ndarray

    def __init__(
        self,
        model: kin.Model,
        data: kin.Data,
        q: np.ndarray,
        copy_data: bool = True,
        forward_kinematics: bool = True,
        collision_model=None,
        collision_data=None,
        default_limits: Optional[List[Limit]] = None,
    ):
        """Initialize configuration.

        Args:
            model: Kinodynamic model.
            data: Data corresponding to the model.
            q: Configuration vector.
            copy_data: If true (default), work on an internal copy of the input
                data. Otherwise, work on the input data directly.
            forward_kinematics: If true (default), compute forward kinematics
                from the q into the internal data.
            collision_model: Not supported by Pinker. Must be ``None``.
            collision_data: Not supported by Pinker. Must be ``None``.
            default_limits: Limits enforced by default when calling
                :func:`.solve_ik` without a custom `limits` keyword argument.
                Defaults to the configuration and velocity limits read from the
                model.

        Notes:
            Configurations copy data and run forward kinematics by default so
            that they are less error-prone for newcomers. You can avoid copies
            or forward kinematics (e.g. if it is already computed by the
            caller) using constructor parameters.
        """
        if collision_model is not None or collision_data is not None:
            raise NotImplementedError(
                "Pinker does not support collision models; "
                "use Pink for collision-aware inverse kinematics"
            )
        q_readonly = q.copy()
        q_readonly.setflags(write=False)
        self.collision_model = None
        self.collision_data = None
        self.data = data.copy() if copy_data else data
        self.default_limits = (
            list(default_limits)
            if default_limits is not None
            else [ConfigurationLimit(model), VelocityLimit(model)]
        )
        self.model = model
        self.q = q_readonly
        self.tangent = VectorSpace(model.nv)

        if forward_kinematics:
            self.update(None)

    def update(self, q: Optional[np.ndarray] = None) -> None:
        """Update configuration to a new vector.

        Calling this function runs forward kinematics.

        Args:
            q: New configuration vector.
        """
        if q is not None:
            q_readonly = q.copy()
            q_readonly.setflags(write=False)
            self.q = q_readonly

        kin.compute_joint_jacobians(self.model, self.data, self.q)
        kin.update_frame_placements(self.model, self.data)

    def check_limits(
        self, tol: float = 1e-6, safety_break: bool = True
    ) -> None:
        """Check that the current configuration is within limits.

        Args:
            tol: Tolerance in radians.
            safety_break: If True, stop execution and raise an exception if the
                current configuration is outside limits. If False, print a
                warning and continue execution.

        Raises:
            NotWithinConfigurationLimits: If the current configuration is
                outside limits.
        """
        q_max = self.model.upper_position_limit
        q_min = self.model.lower_position_limit
        root_nq, _ = self.model.get_root_joint_dim()
        for i in range(root_nq, self.model.nq):
            if q_max[i] <= q_min[i] + tol:  # no limit
                continue
            if self.q[i] < q_min[i] - tol or self.q[i] > q_max[i] + tol:
                if safety_break:
                    raise NotWithinConfigurationLimits(
                        i,
                        self.q[i],
                        q_min[i],
                        q_max[i],
                    )
                logging.warning(
                    "Value %f at index %d is out of limits: [%f, %f]",
                    self.q[i],
                    i,
                    q_min[i],
                    q_max[i],
                )

    def get_frame_jacobian(self, frame: str) -> np.ndarray:
        r"""Compute the Jacobian matrix of a frame velocity.

        Denoting our frame by :math:`B` and the world frame by :math:`W`, the
        Jacobian matrix :math:`{}_B J_{WB}` is related to the body velocity
        :math:`{}_B v_{WB}` by:

        .. math::

            {}_B v_{WB} = {}_B J_{WB} \dot{q}

        Args:
            frame: Name of the frame, typically a link name from the URDF.

        Returns:
            Jacobian :math:`{}_B J_{WB}` of the frame.

        When the robot model includes a floating base
        (kin.JointModelFreeFlyer), the configuration vector :math:`q` consists
        of:

        - ``q[0:3]``: position in [m] of the floating base in the inertial
          frame, formatted as :math:`[p_x, p_y, p_z]`.
        - ``q[3:7]``: unit quaternion for the orientation of the floating base
          in the inertial frame, formatted as :math:`[q_x, q_y, q_z, q_w]`.
        - ``q[7:]``: joint angles in [rad].
        """
        if not self.model.exist_frame(frame):
            raise FrameNotFound(frame, self.model.frames)
        frame_id = self.model.get_frame_id(frame)
        J: np.ndarray = kin.get_frame_jacobian(
            self.model, self.data, frame_id, kin.ReferenceFrame.LOCAL
        )
        return J

    def get_transform_frame_to_world(self, frame: str) -> kin.SE3:
        """Get the pose of a frame in the current configuration.

        Args:
            frame: Name of a frame, typically a link name from the URDF.

        Returns:
            Current transform from the given frame to the world frame.

        Raises:
            FrameNotFound: if the frame name is not found in the robot model.
        """
        if not self.model.exist_frame(frame):
            raise FrameNotFound(frame, self.model.frames)
        frame_id = self.model.get_frame_id(frame)
        return self.data.oMf[frame_id].copy()

    def get_transform(self, source: str, dest: str) -> kin.SE3:
        """Get the pose of a frame with respect to another frame.

        Args:
            source: Name of the frame to get the pose of.
            dest: Name of the frame to get the pose in.

        Returns:
            Current transform from the source frame to the dest frame.

        Raises:
            FrameNotFound: if any frame name is not found in the model.
        """
        transform_source_to_world = self.get_transform_frame_to_world(source)
        transform_dest_to_world = self.get_transform_frame_to_world(dest)
        return transform_dest_to_world.act_inv(transform_source_to_world)

    def integrate(self, velocity, dt) -> "Configuration":
        """Integrate a velocity starting from the current configuration.

        Args:
            velocity: Velocity in tangent space.
            dt: Integration duration in [s].

        Returns:
            New configuration after integration, with the same default limits
            as this one.
        """
        q = kin.integrate(self.model, self.q, velocity * dt)
        return Configuration(
            self.model, self.data, q, default_limits=self.default_limits
        )

    def integrate_inplace(self, velocity, dt) -> None:
        """Integrate a velocity starting from the current configuration.

        Args:
            velocity: Velocity in tangent space.
            dt: Integration duration in [s].
        """
        q = kin.integrate(self.model, self.q, velocity * dt)
        self.update(q)
