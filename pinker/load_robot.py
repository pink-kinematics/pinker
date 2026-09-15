# SPDX-License-Identifier: Apache-2.0

"""Load a robot model from a URDF file or from a robot description."""

import os
from typing import List, Optional, Union

from .kinematics import (
    JointModel,
    JointModelFreeFlyer,
    JointModelPlanar,
    JointModelSpherical,
    RobotWrapper,
)

ROOT_JOINTS = {
    "free_flyer": JointModelFreeFlyer,
    "planar": JointModelPlanar,
    "spherical": JointModelSpherical,
}
"""Root joints that can be named by string, for the ``root_joint`` argument."""


def __make_root_joint(
    root_joint: Optional[Union[str, JointModel]],
) -> Optional[JointModel]:
    """Turn a root-joint name into a joint model.

    Args:
        root_joint: Name of a root joint, joint model, or None.

    Returns:
        Corresponding joint model, or None for a fixed root.

    Raises:
        ValueError: if the root joint is not a known name.
    """
    if root_joint is None or isinstance(root_joint, JointModel):
        return root_joint
    if not isinstance(root_joint, str) or root_joint not in ROOT_JOINTS:
        raise ValueError(
            f"Unknown root joint {root_joint!r}, "
            f"expected one of {sorted(ROOT_JOINTS)} or a joint model"
        )
    return ROOT_JOINTS[root_joint]()


def load_robot(
    robot: Union[str, os.PathLike],
    root_joint: Optional[Union[str, JointModel]] = None,
    package_dirs: Optional[List[str]] = None,
    commit: Optional[str] = None,
) -> RobotWrapper:
    """Load a robot model, from a URDF file or from a robot description.

    The robot bundles a model with its data and initial configuration, which
    is what a :class:`.Configuration` is built from:

    .. code:: python

        robot = pinker.load_robot("ur3_official_description")
        configuration = pinker.Configuration(
            robot.model, robot.data, robot.q0
        )

    Args:
        robot: Path to a URDF file, or name of a description from
            `robot_descriptions
            <https://github.com/robot-descriptions/robot_descriptions.py>`__
            such as ``"ur3_official_description"``.
        root_joint: Joint between the world and the root link of the robot,
            either by name (``"free_flyer"``, ``"planar"`` or
            ``"spherical"``) or as a joint model. Defaults to a fixed root.
        package_dirs: Directories where mesh files are looked up. Only
            applies when loading a URDF file; descriptions locate their own
            meshes.
        commit: If specified, check out that commit from the cloned robot
            description repository. Only applies to descriptions.

    Returns:
        Robot model, bundled with its data and initial configuration.

    Raises:
        ValueError: if an argument does not apply to the robot being loaded.
    """
    joint = __make_root_joint(root_joint)
    path = os.fspath(robot)
    if path.lower().endswith(".urdf") or os.path.exists(path):
        if commit is not None:
            raise ValueError(
                "The 'commit' argument applies to robot descriptions, "
                f"not to the URDF file {path!r}"
            )
        return RobotWrapper.BuildFromURDF(
            path, package_dirs=package_dirs, root_joint=joint
        )
    if package_dirs is not None:
        raise ValueError(
            "The 'package_dirs' argument applies to URDF files, not to the "
            f"{path!r} robot description, which locates its own meshes"
        )
    from pinker.kinematics.robot_descriptions import load_robot_description

    try:
        return load_robot_description(path, root_joint=joint, commit=commit)
    except ModuleNotFoundError as exc:
        if exc.name == "robot_descriptions":
            raise ModuleNotFoundError(
                f"Loading the {path!r} robot description requires the "
                "robot_descriptions package: pip install robot_descriptions"
            ) from exc
        raise
