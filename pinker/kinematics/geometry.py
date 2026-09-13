# SPDX-License-Identifier: Apache-2.0

"""Visual geometry attached to a robot model.

Each geometry object carries a primitive shape (box, sphere, cylinder) or a
mesh file path, together with its placement in the frame of its parent joint.
"""

from typing import List, Optional

import numpy as np

from .se3 import SE3

DEFAULT_COLOR = np.array([0.7, 0.7, 0.7, 1.0])


class GeometryObject:
    """One displayable shape, attached to a joint of the model.

    Attributes:
        name: Name of the geometry object, unique in its geometry model.
        parent_joint: Index of the supporting joint.
        placement: Pose of the shape in the parent joint frame.
        shape: One of "box", "sphere", "cylinder" or "mesh".
        size: Shape parameters: full extents (3,) for a box, (radius,)
            for a sphere, (radius, length) for a cylinder, unused for a
            mesh.
        mesh_path: Path to the mesh file, for mesh shapes.
        mesh_scale: Scale applied to the mesh (3,).
        mesh_color: RGBA color in [0, 1].
    """

    def __init__(
        self,
        name: str,
        parent_joint: int,
        placement: SE3,
        shape: str,
        size: Optional[np.ndarray] = None,
        mesh_path: str = "",
        mesh_scale: Optional[np.ndarray] = None,
        mesh_color: Optional[np.ndarray] = None,
    ):
        """Initialize geometry object.

        Args:
            name: Name of the geometry object, unique in its geometry model.
            parent_joint: Index of the supporting joint.
            placement: Pose of the shape in the parent joint frame.
            shape: One of "box", "sphere", "cylinder" or "mesh".
            size: Shape parameters: full extents (3,) for a box, (radius,)
                for a sphere, (radius, length) for a cylinder, unused for a
                mesh.
            mesh_path: Path to the mesh file, for mesh shapes.
            mesh_scale: Scale applied to the mesh (3,), defaults to ones.
            mesh_color: RGBA color in [0, 1], defaults to light gray.
        """
        self.name = name
        self.parent_joint = parent_joint
        self.placement = placement
        self.shape = shape
        self.size = np.zeros(0) if size is None else np.asarray(size)
        self.mesh_path = mesh_path
        self.mesh_scale = (
            np.ones(3) if mesh_scale is None else np.asarray(mesh_scale)
        )
        self.mesh_color = (
            DEFAULT_COLOR.copy()
            if mesh_color is None
            else np.asarray(mesh_color)
        )

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Human-readable representation of the geometry object.
        """
        return f"GeometryObject({self.name!r}, {self.shape})"


class GeometryModel:
    """Collection of geometry objects attached to a robot model."""

    def __init__(self):
        """Initialize an empty geometry model."""
        self.geometry_objects: List[GeometryObject] = []

    @property
    def ngeoms(self) -> int:
        """Number of geometry objects.

        Returns:
            Number of geometry objects in the model.
        """
        return len(self.geometry_objects)

    def add_geometry_object(self, geometry_object: GeometryObject) -> int:
        """Add a geometry object to the model.

        Args:
            geometry_object: Geometry object to add.

        Returns:
            Index of the new geometry object in the model.
        """
        self.geometry_objects.append(geometry_object)
        return len(self.geometry_objects) - 1
