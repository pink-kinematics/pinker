# SPDX-License-Identifier: Apache-2.0

"""Build a model from a URDF file.

The conventions (joint ordering, frame ordering, default limits) follow
Pinocchio's URDF parser so that models built by pinker.kinematics match those
built by Pinocchio joint by joint and frame by frame:

- Joints are added by depth-first traversal of the kinematic tree, with
  child joints visited in alphabetical order of joint name.
- Each movable joint adds a JOINT frame and a BODY frame for its child
  link. Fixed joints add a FIXED_JOINT frame and a BODY frame attached to
  the nearest movable ancestor joint, and their child link's inertia is
  merged into that joint.
- Continuous joints have their (cos, sin) configuration bounded by 1.01,
  like free-flyer and planar quaternion/cosine-sine coordinates.
"""

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from .joints import (
    JT_FREEFLYER,
    JT_PLANAR,
    JT_PRISMATIC,
    JT_REVOLUTE,
    JT_REVOLUTE_UNBOUNDED,
    JointModel,
)
from .model import Frame, FrameType, Model
from .se3 import SE3, Inertia
from .so3 import rpy_to_matrix

if TYPE_CHECKING:
    from .geometry import GeometryModel


def _parse_origin(element: Optional[ET.Element]) -> SE3:
    """Parse an <origin xyz rpy> element into a transform.

    Args:
        element: Origin element, or None if the URDF has none.

    Returns:
        Transform described by the element, identity if it is None.
    """
    if element is None:
        return SE3.Identity()
    xyz = [float(x) for x in element.get("xyz", "0 0 0").split()]
    rpy = [float(x) for x in element.get("rpy", "0 0 0").split()]
    return SE3(rpy_to_matrix(*rpy), np.array(xyz))


def _parse_inertial(link: ET.Element) -> Inertia:
    """Parse the <inertial> element of a link.

    Args:
        link: Link element to read the inertia from.

    Returns:
        Inertia of the link in its own frame, zero if the link has no
        inertial element.
    """
    inertial = link.find("inertial")
    if inertial is None:
        return Inertia.Zero()
    origin = _parse_origin(inertial.find("origin"))
    mass_elem = inertial.find("mass")
    mass = float(mass_elem.get("value", "0")) if mass_elem is not None else 0.0
    inertia_elem = inertial.find("inertia")
    if inertia_elem is not None:
        ixx = float(inertia_elem.get("ixx", "0"))
        ixy = float(inertia_elem.get("ixy", "0"))
        ixz = float(inertia_elem.get("ixz", "0"))
        iyy = float(inertia_elem.get("iyy", "0"))
        iyz = float(inertia_elem.get("iyz", "0"))
        izz = float(inertia_elem.get("izz", "0"))
        I_com = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    else:
        I_com = np.zeros((3, 3))
    R = origin.rotation
    return Inertia(mass, origin.translation, R @ I_com @ R.T)


class _UrdfJoint:
    """Raw description of a URDF joint element."""

    def __init__(self, element: ET.Element):
        """Read a joint element of the URDF.

        Args:
            element: Joint element to read.
        """
        self.name = element.get("name")
        self.type = element.get("type")
        self.parent = element.find("parent").get("link")
        self.child = element.find("child").get("link")
        self.origin = _parse_origin(element.find("origin"))
        axis_elem = element.find("axis")
        axis = (
            np.array([float(x) for x in axis_elem.get("xyz").split()])
            if axis_elem is not None
            else np.array([1.0, 0.0, 0.0])
        )
        norm = np.linalg.norm(axis)
        self.axis = axis / norm if norm > 0.0 else axis
        self.limit = element.find("limit")

    def limit_value(self, key: str, default: float) -> float:
        """Value of a <limit> attribute, or a default if unspecified.

        Args:
            key: Name of the attribute, e.g. "lower" or "velocity".
            default: Value to return if the attribute is not specified.

        Returns:
            Value of the attribute, or the default.
        """
        if self.limit is None:
            return default
        value = self.limit.get(key)
        return float(value) if value is not None else default


def build_model_from_xml(
    xml_string: str, root_joint: Optional[JointModel] = None
) -> Model:
    """Build a model from a URDF string.

    Args:
        xml_string: URDF description of the robot.
        root_joint: Optional joint model connecting the root link to the
            world, e.g. ``JointModelFreeFlyer()`` for mobile robots.

    Returns:
        Robot model.
    """
    robot = ET.fromstring(xml_string)
    model = Model()
    model.name = robot.get("name", "")

    links: Dict[str, ET.Element] = {}
    for link in robot.findall("link"):
        links[link.get("name")] = link

    # Child joints are visited in alphabetical order of joint name, like
    # Pinocchio (whose urdfdom parser stores joints in a sorted map).
    joints = [_UrdfJoint(element) for element in robot.findall("joint")]
    joints.sort(key=lambda joint: joint.name)
    children: Dict[str, List[_UrdfJoint]] = {name: [] for name in links}
    child_links = set()
    for joint in joints:
        children[joint.parent].append(joint)
        child_links.add(joint.child)
    roots = [name for name in links if name not in child_links]
    if len(roots) != 1:
        raise ValueError(f"expected a single root link, found {roots}")
    root_link = roots[0]

    inf = np.inf
    if root_joint is not None:
        # Pinocchio's addJoint defaults: limits are the largest double, not
        # infinity nor the 1.01 bounds used for URDF floating joints.
        dbl_max = np.finfo(np.float64).max
        root_id = model.add_joint(
            0,
            root_joint,
            SE3.Identity(),
            "root_joint",
            max_effort=np.full(root_joint.nv, dbl_max),
            max_velocity=np.full(root_joint.nv, dbl_max),
            min_config=np.full(root_joint.nq, -dbl_max),
            max_config=np.full(root_joint.nq, dbl_max),
        )
        joint_frame = model.add_joint_frame(root_id)
        model.append_body_to_joint(root_id, _parse_inertial(links[root_link]))
        root_frame = model.add_body_frame(
            root_link, root_id, SE3.Identity(), joint_frame
        )
        root_entry = (root_link, root_id, SE3.Identity(), root_frame)
    else:
        model.append_body_to_joint(0, _parse_inertial(links[root_link]))
        root_frame = model.add_body_frame(root_link, 0, SE3.Identity(), 0)
        root_entry = (root_link, 0, SE3.Identity(), root_frame)

    # Depth-first traversal with child joints in file order, like Pinocchio.
    def visit(link_name, support_joint, link_placement, link_frame) -> None:
        """Add the subtree of a link to the model.

        Args:
            link_name: Name of the link to visit.
            support_joint: Index of the joint supporting the link.
            link_placement: Pose of the link in its support joint frame.
            link_frame: Index of the frame of the link in the model.
        """
        for joint in children[link_name]:
            placement = link_placement * joint.origin
            child_inertia = _parse_inertial(links[joint.child])
            if joint.type == "fixed":
                fixed_frame = model.add_frame(
                    Frame(
                        joint.name,
                        support_joint,
                        link_frame,
                        placement,
                        FrameType.FIXED_JOINT,
                    )
                )
                model.append_body_to_joint(
                    support_joint, child_inertia, placement
                )
                body_frame = model.add_body_frame(
                    joint.child, support_joint, placement, fixed_frame
                )
                visit(joint.child, support_joint, placement, body_frame)
                continue
            if joint.type in ("revolute", "prismatic"):
                jtype = (
                    JT_REVOLUTE if joint.type == "revolute" else JT_PRISMATIC
                )
                joint_model = JointModel(jtype, joint.axis)
                min_config = np.array([joint.limit_value("lower", -inf)])
                max_config = np.array([joint.limit_value("upper", inf)])
            elif joint.type == "continuous":
                joint_model = JointModel(JT_REVOLUTE_UNBOUNDED, joint.axis)
                min_config = np.array([-1.01, -1.01])
                max_config = np.array([1.01, 1.01])
            elif joint.type == "planar":
                joint_model = JointModel(JT_PLANAR)
                min_config = np.array([-inf, -inf, -1.01, -1.01])
                max_config = np.array([inf, inf, 1.01, 1.01])
            elif joint.type == "floating":
                joint_model = JointModel(JT_FREEFLYER)
                min_config = _root_min_config(joint_model)
                max_config = _root_max_config(joint_model)
            else:
                raise ValueError(
                    f"unsupported joint type '{joint.type}' "
                    f"for joint '{joint.name}'"
                )
            max_velocity = np.full(
                joint_model.nv, joint.limit_value("velocity", inf)
            )
            max_effort = np.full(
                joint_model.nv, joint.limit_value("effort", inf)
            )
            joint_id = model.add_joint(
                support_joint,
                joint_model,
                placement,
                joint.name,
                max_effort=max_effort,
                max_velocity=max_velocity,
                min_config=min_config,
                max_config=max_config,
            )
            joint_frame = model.add_joint_frame(joint_id, link_frame)
            model.append_body_to_joint(joint_id, child_inertia)
            body_frame = model.add_body_frame(
                joint.child, joint_id, SE3.Identity(), joint_frame
            )
            visit(joint.child, joint_id, SE3.Identity(), body_frame)

    visit(*root_entry)
    return model


def _root_min_config(joint_model: JointModel) -> np.ndarray:
    """Lower configuration limits of a root joint.

    Args:
        joint_model: Root joint model.

    Returns:
        Lower configuration limits of the root joint.
    """
    inf = np.inf
    if joint_model.jtype == JT_FREEFLYER:
        return np.hstack([np.full(3, -inf), np.full(4, -1.01)])
    if joint_model.jtype == JT_PLANAR:
        return np.array([-inf, -inf, -1.01, -1.01])
    if joint_model.jtype == JT_REVOLUTE_UNBOUNDED:
        return np.array([-1.01, -1.01])
    return np.full(joint_model.nq, -inf)


def _root_max_config(joint_model: JointModel) -> np.ndarray:
    """Upper configuration limits of a root joint.

    Args:
        joint_model: Root joint model.

    Returns:
        Upper configuration limits of the root joint.
    """
    return -_root_min_config(joint_model)


def build_model_from_urdf(
    filename: str, root_joint: Optional[JointModel] = None
) -> Model:
    """Build a model from a URDF file.

    Args:
        filename: Path to the URDF file.
        root_joint: Optional joint model connecting the root link to the
            world.

    Returns:
        Robot model.
    """
    with open(filename, "r", encoding="utf-8") as handle:
        return build_model_from_xml(handle.read(), root_joint)


def _resolve_mesh_path(
    filename: str, urdf_dir: str, package_dirs: List[str]
) -> str:
    """Resolve a URDF mesh filename to a file on disk.

    Args:
        filename: Mesh filename, possibly with a package:// prefix.
        urdf_dir: Directory of the URDF file, for relative paths.
        package_dirs: Directories where ROS packages are looked up.

    Returns:
        Path to the mesh file, or an empty string if not found.
    """
    import os

    if filename.startswith("package://"):
        relative = filename[len("package://") :]
    elif filename.startswith("file://"):
        relative = filename[len("file://") :]
        if os.path.exists(relative):
            return relative
    else:
        relative = filename
        candidate = os.path.join(urdf_dir, relative)
        if os.path.exists(candidate):
            return candidate
        if os.path.exists(relative):
            return relative
    for package_dir in package_dirs:
        candidate = os.path.join(package_dir, relative)
        if os.path.exists(candidate):
            return candidate
        # Also try after dropping the package name from the path
        parts = relative.split("/", 1)
        if len(parts) == 2:
            candidate = os.path.join(package_dir, parts[1])
            if os.path.exists(candidate):
                return candidate
    return ""


def build_geom_from_urdf(
    model: Model,
    filename: str,
    package_dirs: Optional[List[str]] = None,
) -> "GeometryModel":
    """Parse the visual geometries of a URDF file.

    Args:
        model: Model previously built from the same URDF.
        filename: Path to the URDF file.
        package_dirs: Directories where mesh files are looked up.

    Returns:
        Geometry model with one object per URDF <visual> element.
    """
    import os

    from .geometry import GeometryModel, GeometryObject
    from .model import FrameType

    with open(filename, "r", encoding="utf-8") as handle:
        robot = ET.fromstring(handle.read())
    urdf_dir = os.path.dirname(os.path.abspath(filename))
    if package_dirs is None:
        package_dirs = []

    # Named materials defined at the top level of the URDF
    materials = {}
    for material in robot.findall("material"):
        color = material.find("color")
        if color is not None:
            rgba = [float(x) for x in color.get("rgba").split()]
            materials[material.get("name")] = np.array(rgba)

    # BODY frames map each link to its supporting joint and placement
    link_frames = {
        frame.name: frame
        for frame in model.frames
        if frame.type == FrameType.BODY
    }

    geometry_model = GeometryModel()
    for link in robot.findall("link"):
        link_name = link.get("name")
        if link_name not in link_frames:
            continue
        frame = link_frames[link_name]
        for index, visual in enumerate(link.findall("visual")):
            origin = _parse_origin(visual.find("origin"))
            placement = frame.placement * origin
            name = f"{link_name}_{index}"
            color = None
            material = visual.find("material")
            if material is not None:
                color_elem = material.find("color")
                if color_elem is not None:
                    color = np.array(
                        [float(x) for x in color_elem.get("rgba").split()]
                    )
                elif material.get("name") in materials:
                    color = materials[material.get("name")]
            geometry = visual.find("geometry")
            box = geometry.find("box")
            sphere = geometry.find("sphere")
            cylinder = geometry.find("cylinder")
            mesh = geometry.find("mesh")
            if box is not None:
                size = np.array([float(x) for x in box.get("size").split()])
                obj = GeometryObject(
                    name,
                    frame.parent_joint,
                    placement,
                    "box",
                    size,
                    mesh_color=color,
                )
            elif sphere is not None:
                obj = GeometryObject(
                    name,
                    frame.parent_joint,
                    placement,
                    "sphere",
                    np.array([float(sphere.get("radius"))]),
                    mesh_color=color,
                )
            elif cylinder is not None:
                size = np.array(
                    [
                        float(cylinder.get("radius")),
                        float(cylinder.get("length")),
                    ]
                )
                obj = GeometryObject(
                    name,
                    frame.parent_joint,
                    placement,
                    "cylinder",
                    size,
                    mesh_color=color,
                )
            elif mesh is not None:
                mesh_path = _resolve_mesh_path(
                    mesh.get("filename"), urdf_dir, package_dirs
                )
                if not mesh_path:
                    continue  # mesh file not found: skip this visual
                scale = np.array(
                    [float(x) for x in mesh.get("scale", "1 1 1").split()]
                )
                obj = GeometryObject(
                    name,
                    frame.parent_joint,
                    placement,
                    "mesh",
                    mesh_path=mesh_path,
                    mesh_scale=scale,
                    mesh_color=color,
                )
            else:  # unknown geometry type: skip
                continue
            geometry_model.add_geometry_object(obj)
    return geometry_model
