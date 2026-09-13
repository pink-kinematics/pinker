# SPDX-License-Identifier: Apache-2.0

"""Load robot descriptions from robot_descriptions with pinker.kinematics.

This function mirrors ``robot_descriptions.loaders.pinocchio`` for URDF
descriptions. It requires the ``robot_descriptions`` package.
"""

import os
from importlib import import_module
from typing import Optional

from .joints import JointModel
from .robot_wrapper import RobotWrapper


def load_robot_description(
    description_name: str,
    root_joint: Optional[JointModel] = None,
    commit: Optional[str] = None,
) -> RobotWrapper:
    """Load a robot description with pinker.kinematics.

    Args:
        description_name: Name of the robot description, e.g.
            "upkie_description".
        root_joint: Optional joint connecting the root link to the world.
        commit: If specified, check out that commit from the cloned robot
            description repository.

    Returns:
        Robot wrapper, with its visual model.
    """
    if commit is not None:
        os.environ["ROBOT_DESCRIPTION_COMMIT"] = commit
    try:
        module = import_module(f"robot_descriptions.{description_name}")
    finally:
        if commit is not None:
            os.environ.pop("ROBOT_DESCRIPTION_COMMIT", None)
    if hasattr(module, "URDF_PATH"):
        urdf_path = module.URDF_PATH
    else:  # xacro-backed description
        from robot_descriptions._xacro import get_urdf_path

        urdf_path = get_urdf_path(module)
    package_dirs = [
        module.PACKAGE_PATH,
        module.REPOSITORY_PATH,
        os.path.dirname(module.PACKAGE_PATH),
        os.path.dirname(module.REPOSITORY_PATH),
        os.path.dirname(urdf_path),
    ]
    return RobotWrapper.BuildFromURDF(
        urdf_path, package_dirs=package_dirs, root_joint=root_joint
    )
