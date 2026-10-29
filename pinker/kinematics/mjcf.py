# SPDX-License-Identifier: Apache-2.0

"""Build a model from an MJCF file.

The conventions (joint ordering, frame ordering, default limits) follow
Pinocchio's MJCF parser so that models built by pinker.kinematics match those
built by Pinocchio joint by joint and frame by frame:

- Bodies are visited depth-first in file order. A body with a joint adds the
  joint, a JOINT frame and a BODY frame. A body without joint adds a
  FIXED_JOINT frame named ``<body>_fixed`` and a BODY frame attached to the
  nearest movable ancestor joint, and its inertia is merged into that joint.
- Sites add OP_FRAME frames right after the BODY frame of their body.
- Hinge and slide joints without range are unbounded. Free and ball joints
  have their quaternion coordinates bounded by 1.01, like URDF floating
  joints.

There are two deliberate differences with Pinocchio: free-joint translations
are unbounded, and free joints coincide with the world frame (so that their
coordinates are the pose of their body in the world as in MuJoCo).

The parser supports a subset of MJCF: hinge, slide, ball and free joints,
default classes, all orientation specifiers, explicit inertials and sites.
Unsupported features raise a ValueError.
"""

import logging
import os
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .algorithms import exp3
from .joints import (
    JT_FREEFLYER,
    JT_PRISMATIC,
    JT_REVOLUTE,
    JT_SPHERICAL,
    JointModel,
)
from .model import Frame, FrameType, Model
from .se3 import SE3, Inertia
from .so3 import quaternion_to_matrix

if TYPE_CHECKING:
    from .geometry import GeometryModel

ORIENTATION_KEYS = ("quat", "axisangle", "euler", "xyaxes", "zaxis")
"""Attributes that specify an orientation, at most one per element."""

UNSUPPORTED_TAGS = (
    "attach",
    "composite",
    "flexcomp",
    "frame",
    "include",
    "replicate",
)
"""Elements that would change the kinematic tree if we skipped them."""


def _floats(text: str, n: Optional[int] = None) -> np.ndarray:
    """Parse a whitespace-separated list of numbers.

    Args:
        text: Attribute value to parse.
        n: Expected number of values, if it should be checked.

    Returns:
        Parsed values.

    Raises:
        ValueError: If ``n`` is given and the list has a different length.
    """
    values = np.array([float(x) for x in text.split()])
    if n is not None and values.shape[0] != n:
        raise ValueError(f"expected {n} numbers, got '{text}'")
    return values


def _unit(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector.

    Args:
        vector: Vector to normalize.

    Returns:
        Unit vector with the same direction.

    Raises:
        ValueError: If the vector is zero.
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError(f"cannot normalize zero vector {vector}")
    return vector / norm


class _Mjcf:
    """MJCF document with its compiler settings and default classes.

    Attributes:
        root: Root ``<mujoco>`` element.
        angle_scale: Conversion factor from MJCF angles to radians.
        autolimits: Whether joints with a range are limited by default.
        classes: Attributes of each element type, by default class name.
        eulerseq: Sequence of axes for Euler angles.
        meshdir: Directory of mesh files, relative to the model file.
    """

    root: ET.Element
    angle_scale: float
    autolimits: bool
    classes: Dict[str, Dict[str, Dict[str, str]]]
    eulerseq: str
    meshdir: str

    def __init__(self, xml_string: str):
        """Parse an MJCF document.

        Args:
            xml_string: MJCF description of the model.

        Raises:
            ValueError: If the document uses an unsupported MJCF feature.
        """
        root = ET.fromstring(xml_string)
        if root.tag != "mujoco":
            raise ValueError(f"expected a <mujoco> root, got <{root.tag}>")
        for element in root.iter():
            if element.tag in UNSUPPORTED_TAGS:
                raise ValueError(f"<{element.tag}> elements are not supported")

        compiler: Dict[str, str] = {}
        for element in root.findall("compiler"):
            compiler.update(element.attrib)
        angle = compiler.get("angle", "degree")
        if angle not in ("degree", "radian"):
            raise ValueError(f"unknown compiler angle unit '{angle}'")
        eulerseq = compiler.get("eulerseq", "xyz")
        if len(eulerseq) != 3 or not set(eulerseq) <= set("xyzXYZ"):
            raise ValueError(f"invalid Euler sequence '{eulerseq}'")
        if compiler.get("inertiafromgeom", "auto") == "true":
            raise ValueError("computing inertias from geoms is not supported")

        self.root = root
        self.angle_scale = 1.0 if angle == "radian" else np.pi / 180.0
        self.autolimits = compiler.get("autolimits", "true") == "true"
        self.classes = {"main": {}}
        self.eulerseq = eulerseq
        self.meshdir = compiler.get("meshdir", compiler.get("assetdir", ""))
        for default in root.findall("default"):
            self.__parse_default_class(default, {}, is_top_level=True)

    def __parse_default_class(
        self,
        element: ET.Element,
        parent: Dict[str, Dict[str, str]],
        is_top_level: bool,
    ) -> None:
        """Parse a default class and the classes nested in it.

        Args:
            element: The ``<default>`` element of the class.
            parent: Attributes of the parent class, by element type.
            is_top_level: Whether the class is the top-level default class.
        """
        name = element.get("class", "main" if is_top_level else None)
        if name is None:
            raise ValueError("nested default classes must have a name")
        attributes = {tag: dict(attrib) for tag, attrib in parent.items()}
        for child in element:
            if child.tag != "default":
                attributes.setdefault(child.tag, {}).update(child.attrib)
        self.classes[name] = attributes
        for child in element.findall("default"):
            self.__parse_default_class(child, attributes, is_top_level=False)

    def attributes(
        self, element: ET.Element, childclass: str
    ) -> Dict[str, str]:
        """Attributes of an element, completed by its default class.

        Args:
            element: Element to read attributes from.
            childclass: Default class of the enclosing body.

        Returns:
            Attributes of the element, falling back to those of its class.

        Raises:
            ValueError: If the class of the element is not defined.
        """
        class_name = element.get("class", childclass)
        if class_name not in self.classes:
            raise ValueError(f"unknown default class '{class_name}'")
        attributes = dict(self.classes[class_name].get(element.tag, {}))
        if any(key in element.attrib for key in ORIENTATION_KEYS):
            for key in ORIENTATION_KEYS:  # the element's own takes precedence
                attributes.pop(key, None)
        attributes.update(element.attrib)
        return attributes

    def pose(self, attributes: Dict[str, str]) -> SE3:
        """Pose given by the position and orientation of an element.

        Args:
            attributes: Attributes of the element.

        Returns:
            Pose of the element in the frame of its parent body.

        Raises:
            ValueError: If the element has several orientation specifiers.
        """
        keys = [key for key in ORIENTATION_KEYS if key in attributes]
        if len(keys) > 1:
            raise ValueError(f"conflicting orientation specifiers {keys}")
        translation = _floats(attributes.get("pos", "0 0 0"), 3)
        rotation = np.eye(3)
        if "quat" in attributes:
            w, x, y, z = _unit(_floats(attributes["quat"], 4))
            rotation = quaternion_to_matrix([x, y, z, w])
        elif "axisangle" in attributes:
            axis_angle = _floats(attributes["axisangle"], 4)
            angle = axis_angle[3] * self.angle_scale
            rotation = exp3(_unit(axis_angle[:3]) * angle)
        elif "euler" in attributes:
            angles = _floats(attributes["euler"], 3) * self.angle_scale
            for axis_name, angle in zip(self.eulerseq, angles):
                axis = np.eye(3)["xyz".index(axis_name.lower())]
                R = exp3(axis * angle)
                if axis_name.islower():  # about moving axes, as in MuJoCo
                    rotation = rotation @ R
                else:  # about fixed axes
                    rotation = R @ rotation
        elif "xyaxes" in attributes:
            xyaxes = _floats(attributes["xyaxes"], 6)
            x_axis = _unit(xyaxes[:3])
            y_axis = _unit(xyaxes[3:] - x_axis.dot(xyaxes[3:]) * x_axis)
            rotation = np.column_stack(
                [x_axis, y_axis, np.cross(x_axis, y_axis)]
            )
        elif "zaxis" in attributes:
            rotation = _rotation_from_z(_unit(_floats(attributes["zaxis"], 3)))
        return SE3(rotation, translation)

    def bodies(
        self,
    ) -> Iterator[Tuple[ET.Element, Optional[ET.Element], str, str]]:
        """Iterate over bodies depth-first, in file order.

        Returns:
            Iterator over tuples ``(body, parent, name, childclass)`` where
            ``parent`` is None for top-level bodies and ``childclass`` is the
            default class of the elements of the body.
        """
        names: Dict[ET.Element, str] = {}
        for worldbody in self.root.findall("worldbody"):
            stack: List[Tuple[ET.Element, Optional[ET.Element], str]] = [
                (body, None, worldbody.get("childclass", "main"))
                for body in reversed(worldbody.findall("body"))
            ]
            while stack:
                body, parent, childclass = stack.pop()
                parent_name = names[parent] if parent is not None else ""
                name = body.get("name", f"{parent_name}Bis")
                childclass = body.get("childclass", childclass)
                names[body] = name
                yield body, parent, name, childclass
                for child in reversed(body.findall("body")):
                    stack.append((child, body, childclass))


def _rotation_from_z(z_axis: np.ndarray) -> np.ndarray:
    """Minimal rotation that maps the z-axis to a given unit vector.

    Args:
        z_axis: Unit vector to map the z-axis to.

    Returns:
        Rotation matrix.
    """
    cos_angle = z_axis[2]
    axis = np.cross([0.0, 0.0, 1.0], z_axis)
    sin_angle = np.linalg.norm(axis)
    if sin_angle < 1e-12:  # aligned or opposite
        return np.eye(3) if cos_angle > 0.0 else np.diag([1.0, -1.0, -1.0])
    return exp3(axis / sin_angle * np.arctan2(sin_angle, cos_angle))


def _parse_inertial(mjcf: _Mjcf, body: ET.Element, body_name: str) -> Inertia:
    """Parse the <inertial> element of a body.

    Args:
        mjcf: MJCF document.
        body: Body element to read the inertia from.
        body_name: Name of the body, for error messages.

    Returns:
        Inertia of the body in its own frame, zero if the body has neither
        inertial element nor geoms.

    Raises:
        ValueError: If the inertia of the body would be computed from its
            geoms, which is not supported.
    """
    inertial = body.find("inertial")
    if inertial is None:
        if body.find("geom") is not None:
            raise ValueError(
                f"body '{body_name}' has no <inertial> element: computing "
                "inertias from geoms is not supported"
            )
        return Inertia.Zero()
    mass = float(inertial.get("mass"))
    pose = mjcf.pose(inertial.attrib)
    if "diaginertia" in inertial.attrib:
        I_com = np.diag(_floats(inertial.get("diaginertia"), 3))
    elif "fullinertia" in inertial.attrib:
        ixx, iyy, izz, ixy, ixz, iyz = _floats(inertial.get("fullinertia"), 6)
        I_com = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    else:
        raise ValueError(f"inertial of body '{body_name}' has no inertia")
    R = pose.rotation
    return Inertia(mass, pose.translation, R @ I_com @ R.T)


class _MjcfJoint:
    """Joint of an MJCF body, with attributes completed by defaults."""

    def __init__(
        self,
        mjcf: _Mjcf,
        element: ET.Element,
        childclass: str,
        body_name: str,
        index: int,
    ):
        """Read a joint or freejoint element.

        Args:
            mjcf: MJCF document.
            element: Joint element to read.
            childclass: Default class of the enclosing body.
            body_name: Name of the body the joint belongs to.
            index: Index of the joint in its body, to name unnamed joints.

        Raises:
            ValueError: If the joint has an unsupported type or attribute.
        """
        inf = np.inf
        if element.tag == "freejoint":
            attributes = dict(element.attrib)
            attributes["type"] = "free"
            default_name = f"{body_name}_free"
        else:
            attributes = mjcf.attributes(element, childclass)
            default_name = f"{body_name}Joint_{index}"
        self.name = attributes.get("name", default_name)
        self.type = attributes.get("type", "hinge")
        self.pos = _floats(attributes.get("pos", "0 0 0"), 3)
        self.axis = _unit(_floats(attributes.get("axis", "0 0 1"), 3))
        if float(attributes.get("ref", "0")) != 0.0:
            raise ValueError(
                f"joint '{self.name}' has a reference position, "
                "which is not supported"
            )
        limited = attributes.get("limited", "auto")
        is_limited = (
            mjcf.autolimits and "range" in attributes
            if limited == "auto"
            else limited == "true"
        )
        lower, upper = _floats(attributes.get("range", "0 0"), 2)
        if self.type == "hinge":  # slide ranges are lengths, not angles
            lower *= mjcf.angle_scale
            upper *= mjcf.angle_scale
        if not is_limited:
            lower, upper = -inf, inf
        effort = inf
        if "actuatorfrcrange" in attributes:
            effort = _floats(attributes["actuatorfrcrange"], 2)[1]

        if self.type == "hinge":
            self.model = JointModel(JT_REVOLUTE, self.axis)
        elif self.type == "slide":
            self.model = JointModel(JT_PRISMATIC, self.axis)
        elif self.type == "ball":
            self.model = JointModel(JT_SPHERICAL)
        elif self.type == "free":
            self.model = JointModel(JT_FREEFLYER)
        else:
            raise ValueError(
                f"unsupported joint type '{self.type}' for joint '{self.name}'"
            )
        if self.type in ("hinge", "slide"):
            self.min_config = np.array([lower])
            self.max_config = np.array([upper])
        else:
            if self.type == "ball" and is_limited:
                logging.warning(
                    "Cone limit of ball joint '%s' is not supported",
                    self.name,
                )
            nq_translation = 3 if self.type == "free" else 0
            self.max_config = np.hstack(
                [np.full(nq_translation, inf), np.full(4, 1.01)]
            )
            self.min_config = -self.max_config
        self.max_effort = np.full(self.model.nv, effort)
        self.max_velocity = np.full(self.model.nv, inf)


def build_model_from_mjcf_xml(
    xml_string: str, root_joint: Optional[JointModel] = None
) -> Model:
    """Build a model from an MJCF string.

    Args:
        xml_string: MJCF description of the robot.
        root_joint: Optional joint model connecting the root body to the
            world, e.g. ``JointModelFreeFlyer()`` for mobile robots. It is
            ignored, with a warning, if the root body already has a joint.

    Returns:
        Robot model.

    Raises:
        ValueError: If the description uses an unsupported MJCF feature.
    """
    mjcf = _Mjcf(xml_string)
    model = Model()
    model.name = mjcf.root.get("model", "")

    nb_top_level_bodies = len(mjcf.root.findall("worldbody/body"))
    if root_joint is not None and nb_top_level_bodies != 1:
        raise ValueError(
            "a root joint requires a single top-level body, "
            f"found {nb_top_level_bodies}"
        )
    for worldbody in mjcf.root.findall("worldbody"):
        _add_sites(mjcf, model, worldbody, "world", "main", 0)

    body_frames: Dict[ET.Element, int] = {}
    for body, parent, name, childclass in mjcf.bodies():
        inertia = _parse_inertial(mjcf, body, name)
        joints = [
            _MjcfJoint(mjcf, element, childclass, name, index)
            for index, element in enumerate(
                child for child in body if child.tag in ("joint", "freejoint")
            )
        ]
        if len(joints) > 1:
            raise ValueError(
                f"body '{name}' has {len(joints)} joints: bodies with "
                "several joints are not supported"
            )
        if parent is not None and joints and joints[0].type == "free":
            raise ValueError(
                f"free joint '{joints[0].name}' is not in a top-level body"
            )
        if parent is None and root_joint is not None and joints:
            logging.warning(
                "Ignoring root joint as body '%s' already has joint '%s'",
                name,
                joints[0].name,
            )
        parent_frame_id = body_frames[parent] if parent is not None else 0
        parent_frame = model.frames[parent_frame_id]
        body_pose = parent_frame.placement * mjcf.pose(body.attrib)

        if parent is None and not joints and root_joint is not None:
            # As in Pinocchio, the root joint coincides with the world frame
            joint_id = model.add_joint(
                0, root_joint, SE3.Identity(), "root_joint"
            )
            joint_frame = model.add_joint_frame(joint_id, 0)
            model.append_body_to_joint(joint_id, inertia)
            body_frame = model.add_body_frame(
                name, joint_id, SE3.Identity(), joint_frame
            )
        elif parent is None and not joints:
            model.append_body_to_joint(0, inertia, body_pose)
            body_frame = model.add_body_frame(name, 0, body_pose, 0)
        elif not joints:
            support_joint = parent_frame.parent_joint
            fixed_frame = model.add_frame(
                Frame(
                    f"{name}_fixed",
                    support_joint,
                    parent_frame_id,
                    body_pose,
                    FrameType.FIXED_JOINT,
                )
            )
            model.append_body_to_joint(support_joint, inertia, body_pose)
            body_frame = model.add_body_frame(
                name, support_joint, body_pose, fixed_frame
            )
        else:
            joint = joints[0]
            joint_in_body = SE3(np.eye(3), joint.pos)
            placement = (
                SE3.Identity()
                if joint.type == "free"
                else body_pose * joint_in_body
            )
            body_in_joint = (
                SE3.Identity()
                if joint.type == "free"
                else joint_in_body.inverse()
            )
            joint_id = model.add_joint(
                parent_frame.parent_joint,
                joint.model,
                placement,
                joint.name,
                max_effort=joint.max_effort,
                max_velocity=joint.max_velocity,
                min_config=joint.min_config,
                max_config=joint.max_config,
            )
            joint_frame = model.add_joint_frame(joint_id, parent_frame_id)
            model.append_body_to_joint(joint_id, inertia, body_in_joint)
            body_frame = model.add_body_frame(
                name, joint_id, body_in_joint, joint_frame
            )
        body_frames[body] = body_frame
        _add_sites(mjcf, model, body, name, childclass, body_frame)
    return model


def _add_sites(
    mjcf: _Mjcf,
    model: Model,
    body: ET.Element,
    body_name: str,
    childclass: str,
    body_frame: int,
) -> None:
    """Add the sites of a body to the model as operational frames.

    Args:
        mjcf: MJCF document.
        model: Model to add the frames to.
        body: Body (or worldbody) element whose sites to add.
        body_name: Name of the body, to name unnamed sites.
        childclass: Default class of the elements of the body.
        body_frame: Index of the frame of the body in the model.
    """
    frame = model.frames[body_frame]
    for index, site in enumerate(body.findall("site")):
        attributes = mjcf.attributes(site, childclass)
        model.add_frame(
            Frame(
                attributes.get("name", f"{body_name}Site_{index}"),
                frame.parent_joint,
                body_frame,
                frame.placement * mjcf.pose(attributes),
                FrameType.OP_FRAME,
            )
        )


def build_model_from_mjcf(
    filename: str, root_joint: Optional[JointModel] = None
) -> Model:
    """Build a model from an MJCF file.

    Args:
        filename: Path to the MJCF file.
        root_joint: Optional joint model connecting the root body to the
            world. It is ignored, with a warning, if the root body already
            has a joint, for instance a ``<freejoint>``.

    Returns:
        Robot model.

    Raises:
        ValueError: If the description uses an unsupported MJCF feature.
    """
    with open(filename, "r", encoding="utf-8") as handle:
        return build_model_from_mjcf_xml(handle.read(), root_joint)


def _geom_kind_is_collision(attributes: Dict[str, str]) -> bool:
    """Whether a geom is only meant for collisions, as Pinocchio reads it.

    Args:
        attributes: Attributes of the geom.

    Returns:
        False for geoms that don't collide (visual) or that are in a group
        displayed by default, True for collision-only geoms.
    """
    contype = int(attributes.get("contype", "1"))
    conaffinity = int(attributes.get("conaffinity", "1"))
    if contype == 0 and conaffinity == 0:
        return False
    return int(attributes.get("group", "0")) > 2


def build_geom_from_mjcf(model: Model, filename: str) -> "GeometryModel":
    """Parse the visual geometries of an MJCF file.

    Geoms that don't collide, or that collide but are in a group visible by
    default (0 to 2), are visual geometries. Mesh, box, sphere and cylinder
    geoms are supported, other shapes are skipped.

    Args:
        model: Model previously built from the same MJCF.
        filename: Path to the MJCF file.

    Returns:
        Geometry model with one object per visual geom.
    """
    from .geometry import GeometryModel, GeometryObject

    with open(filename, "r", encoding="utf-8") as handle:
        mjcf = _Mjcf(handle.read())
    mesh_dir = os.path.join(
        os.path.dirname(os.path.abspath(filename)), mjcf.meshdir
    )

    meshes: Dict[str, Tuple[str, np.ndarray]] = {}
    materials: Dict[str, np.ndarray] = {}
    for asset in mjcf.root.findall("asset"):
        for mesh in asset.findall("mesh"):
            attributes = mjcf.attributes(mesh, "main")
            if "file" not in attributes:
                continue  # mesh defined by vertices: skip it
            file = attributes["file"]
            name = attributes.get(
                "name", os.path.splitext(os.path.basename(file))[0]
            )
            scale = _floats(attributes.get("scale", "1 1 1"), 3)
            meshes[name] = (os.path.join(mesh_dir, file), scale)
        for material in asset.findall("material"):
            attributes = mjcf.attributes(material, "main")
            materials[attributes["name"]] = _floats(
                attributes.get("rgba", "1 1 1 1"), 4
            )

    body_frames = {
        frame.name: frame
        for frame in model.frames
        if frame.type == FrameType.BODY
    }
    world_frame = model.frames[0]
    elements = [
        (worldbody, "world", "main", world_frame)
        for worldbody in mjcf.root.findall("worldbody")
    ]
    elements.extend(
        (body, name, childclass, body_frames[name])
        for body, _, name, childclass in mjcf.bodies()
        if name in body_frames
    )

    geometry_model = GeometryModel()
    for body, body_name, childclass, frame in elements:
        for index, geom in enumerate(body.findall("geom")):
            attributes = mjcf.attributes(geom, childclass)
            if _geom_kind_is_collision(attributes):
                continue
            name = attributes.get("name", f"{body_name}_{index}")
            shape = attributes.get("type", "sphere")
            size = _floats(attributes.get("size", "0 0 0"))
            if "fromto" in attributes and shape == "cylinder":
                fromto = _floats(attributes["fromto"], 6)
                start, end = fromto[:3], fromto[3:]
                rotation = _rotation_from_z(_unit(end - start))
                pose = SE3(rotation, 0.5 * (start + end))
                size = np.array([size[0], 0.5 * np.linalg.norm(end - start)])
            elif "fromto" in attributes:
                continue  # unsupported shape anyway
            else:
                pose = mjcf.pose(attributes)
            placement = frame.placement * pose
            color = None
            if attributes.get("material") in materials:
                color = materials[attributes["material"]]
            elif "rgba" in attributes:
                color = _floats(attributes["rgba"], 4)
            if shape == "box":
                obj = GeometryObject(
                    name,
                    frame.parent_joint,
                    placement,
                    "box",
                    2.0 * size[:3],  # MJCF sizes are half extents
                    mesh_color=color,
                )
            elif shape == "sphere":
                obj = GeometryObject(
                    name,
                    frame.parent_joint,
                    placement,
                    "sphere",
                    size[:1],
                    mesh_color=color,
                )
            elif shape == "cylinder":
                obj = GeometryObject(
                    name,
                    frame.parent_joint,
                    placement,
                    "cylinder",
                    np.array([size[0], 2.0 * size[1]]),  # half length
                    mesh_color=color,
                )
            elif shape == "mesh" and attributes.get("mesh") in meshes:
                mesh_path, scale = meshes[attributes["mesh"]]
                if not os.path.exists(mesh_path):
                    continue  # mesh file not found: skip this geom
                obj = GeometryObject(
                    name,
                    frame.parent_joint,
                    placement,
                    "mesh",
                    mesh_path=mesh_path,
                    mesh_scale=scale,
                    mesh_color=color,
                )
            else:  # unsupported geometry type: skip
                continue
            geometry_model.add_geometry_object(obj)
    return geometry_model
