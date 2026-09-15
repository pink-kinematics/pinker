# SPDX-License-Identifier: Apache-2.0

"""Load robot models from robot descriptions or from URDF files."""

import os
from importlib import import_module
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


def load_robot_description(
    description_name: str,
    root_joint: Optional[Union[str, JointModel]] = None,
    commit: Optional[str] = None,
) -> RobotWrapper:
    """Load a robot description with Pinker's kinematics.

    Args:
        description_name: Name of the robot description, for instance
            ``"ur3_official_description"``.
        root_joint: Joint between the world and the root link of the robot,
            either by name (``"free_flyer"``, ``"planar"`` or
            ``"spherical"``) or as a joint model. Defaults to a fixed root.
        commit: If specified, check out that commit from the cloned robot
            description repository.

    Returns:
        Robot model, bundled with its data and initial configuration.

    Raises:
        ModuleNotFoundError: if the ``robot_descriptions`` package is not
            installed, or if it has no such description.
        ValueError: if the root joint is not a known name.
    """
    joint = __make_root_joint(root_joint)
    if commit is not None:
        os.environ["ROBOT_DESCRIPTION_COMMIT"] = commit
    try:
        module = import_module(f"robot_descriptions.{description_name}")
    except ModuleNotFoundError as exc:
        if exc.name == "robot_descriptions":
            raise ModuleNotFoundError(
                f"Loading the {description_name!r} robot description requires"
                " the robot_descriptions package:"
                " pip install robot_descriptions"
            ) from exc
        raise
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
        urdf_path, package_dirs=package_dirs, root_joint=joint
    )


def load_robot_urdf(
    urdf_path: Union[str, os.PathLike],
    root_joint: Optional[Union[str, JointModel]] = None,
    package_dirs: Optional[List[str]] = None,
) -> RobotWrapper:
    """Load a robot model from a URDF file.

    The robot bundles a model with its data and initial configuration, which
    is what a :class:`.Configuration` is built from:

    .. code:: python

        robot = pinker.load_robot_urdf("robot.urdf")
        configuration = pinker.Configuration(
            robot.model, robot.data, robot.q0
        )

    Args:
        urdf_path: Path to the URDF file.
        root_joint: Joint between the world and the root link of the robot,
            either by name (``"free_flyer"``, ``"planar"`` or
            ``"spherical"``) or as a joint model. Defaults to a fixed root.
        package_dirs: Directories where mesh files are looked up. Mesh files
            are only read when the robot is displayed, so this argument can be
            left out when it is not.

    Returns:
        Robot model, bundled with its data and initial configuration.

    Raises:
        FileNotFoundError: if there is no URDF file at that path.
        ValueError: if the root joint is not a known name.
    """
    joint = __make_root_joint(root_joint)
    return RobotWrapper.BuildFromURDF(
        os.fspath(urdf_path), package_dirs=package_dirs, root_joint=joint
    )
