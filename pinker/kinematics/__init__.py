# SPDX-License-Identifier: Apache-2.0

"""Kinematics backend of Pinker.

This module implements the kinematics functions used by Pinker:

- models built from URDF (revolute, continuous, prismatic, planar, floating and
  fixed joints);
- forward kinematics, frame placements, frame and joint Jacobians in world,
  local and local-world-aligned frames;
- Lie-group operations on SE(3) and configuration spaces;
- center of mass and its Jacobian.

Heavy computations are delegated to an internal C extension. Functions were
implemented based on their homonyms in `Pinocchio v4.1.0
<https://github.com/stack-of-tasks/pinocchio/tree/v4.1.0>`__ as a reference.
Their outputs are cross-validated against Pinocchio by the test suite in
`tests/kinematics`.
"""

__version__ = "0.1.0"

from .algorithms import (
    ARG0,
    ARG1,
    Jlog6,
    center_of_mass,
    compute_joint_jacobians,
    custom_configuration,
    d_difference,
    difference,
    exp3,
    exp6,
    forward_kinematics,
    frames_forward_kinematics,
    get_frame_jacobian,
    get_joint_jacobian,
    integrate,
    jacobian_center_of_mass,
    log3,
    log6,
    neutral,
    update_frame_placements,
)
from .geometry import GeometryModel, GeometryObject
from .joints import (
    JointModel,
    JointModelFreeFlyer,
    JointModelPlanar,
    JointModelPrismaticUnaligned,
    JointModelPX,
    JointModelPY,
    JointModelPZ,
    JointModelRevoluteUnaligned,
    JointModelRX,
    JointModelRY,
    JointModelRZ,
    JointModelSpherical,
)
from .model import Data, Frame, FrameType, Model, ReferenceFrame
from .robot_wrapper import RobotWrapper
from .se3 import SE3, Inertia, Motion
from .so3 import (
    quaternion_to_matrix,
    quaternion_wxyz,
    rpy_to_matrix,
    skew,
)
from .urdf import (
    build_geom_from_urdf,
    build_model_from_urdf,
    build_model_from_xml,
)

__all__ = [
    "ARG0",
    "ARG1",
    "Data",
    "Frame",
    "FrameType",
    "GeometryModel",
    "GeometryObject",
    "Inertia",
    "Jlog6",
    "JointModel",
    "JointModelFreeFlyer",
    "JointModelPX",
    "JointModelPY",
    "JointModelPZ",
    "JointModelPlanar",
    "JointModelPrismaticUnaligned",
    "JointModelRX",
    "JointModelRY",
    "JointModelRZ",
    "JointModelRevoluteUnaligned",
    "JointModelSpherical",
    "Model",
    "Motion",
    "ReferenceFrame",
    "RobotWrapper",
    "SE3",
    "build_geom_from_urdf",
    "build_model_from_urdf",
    "build_model_from_xml",
    "center_of_mass",
    "compute_joint_jacobians",
    "custom_configuration",
    "d_difference",
    "difference",
    "exp3",
    "exp6",
    "forward_kinematics",
    "frames_forward_kinematics",
    "get_frame_jacobian",
    "get_joint_jacobian",
    "integrate",
    "jacobian_center_of_mass",
    "log3",
    "log6",
    "neutral",
    "quaternion_to_matrix",
    "quaternion_wxyz",
    "rpy_to_matrix",
    "skew",
    "update_frame_placements",
]
