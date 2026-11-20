# SPDX-License-Identifier: Apache-2.0

"""Robot visualization with Viser."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, Optional, Sequence

import numpy as np

from . import kinematics as kin
from .kinematics import GeometryModel, GeometryObject, GeometryType
from .kinematics.model import Data, Model
from .kinematics.so3 import quaternion_wxyz

if TYPE_CHECKING:  # viser is imported lazily, when starting the viewer
    import viser


class ViserVisualizer:
    """Robot visualizer using Viser.

    The viewer and the scene nodes are created when the visualizer starts:
    :func:`init_viewer` connects to a Viser server, then
    :func:`load_viewer_model` adds the robot's visuals and frames to its
    scene.

    Attributes:
        collision_model: Geometry model with the robot's collision shapes.
        collision_root_frame: Scene node the collision shapes of the model
            hang from.
        collision_root_node_name: Name of that scene node.
        data: Data where the visualizer runs its forward kinematics.
        frames: Scene nodes of the robot's visuals and frames, by node name.
        frames_root_frame: Scene node the frames of the model hang from.
        frames_root_node_name: Name of that scene node.
        model: Robot model to visualize.
        viewer: Viser server the scene is displayed in, None until
            :func:`init_viewer` is called.
        viewer_root_node_name: Name of the scene node the robot hangs from.
        visual_model: Geometry model with the robot's visuals.
        visual_root_frame: Scene node the visuals of the model hang from.
        visual_root_node_name: Name of that scene node.
    """

    collision_model: Optional[GeometryModel]
    collision_root_frame: viser.FrameHandle
    collision_root_node_name: str
    data: Data
    frames: Dict[str, viser.SceneNodeHandle]
    frames_root_frame: viser.FrameHandle
    frames_root_node_name: str
    model: Model
    viewer: Optional[viser.ViserServer]
    viewer_root_node_name: str
    visual_model: Optional[GeometryModel]
    visual_root_frame: viser.FrameHandle
    visual_root_node_name: str

    def __init__(
        self,
        model: Model,
        visual_model: Optional[GeometryModel] = None,
        collision_model: Optional[GeometryModel] = None,
    ):
        """Initialize visualizer from a model and its geometry models.

        Args:
            model: Robot model to visualize.
            visual_model: Geometry model with the robot's visuals.
            collision_model: Geometry model with the robot's collision shapes.
        """
        self.model = model
        self.visual_model = visual_model
        self.collision_model = collision_model
        self.data: Data = model.create_data()
        self.viewer = None
        self.frames = {}

    @property
    def _scene(self) -> viser.SceneApi:
        """Scene the robot is displayed in.

        Returns:
            Scene of the viewer.

        Raises:
            RuntimeError: If the viewer has not been started yet.
        """
        if self.viewer is None:
            raise RuntimeError(
                "there is no viewer to display in yet, call init_viewer first"
            )
        return self.viewer.scene

    def init_viewer(
        self,
        viewer=None,
        open: bool = False,
        loadModel: bool = False,
        host: str = "localhost",
        port: int = 8080,
    ) -> None:
        """Start a new Viser server (or reuse the one passed as viewer).

        Args:
            viewer: Existing Viser server to reuse. If None, a new server is
                started.
            open: If true, open the viewer in a web browser and wait for a
                client to connect, for at most ten seconds.
            loadModel: If true, load the visual model in the viewer.
            host: Host name the new server listens to.
            port: Port the new server listens to.
        """
        import viser

        self.viewer = viewer or viser.ViserServer(host=host, server_port=port)
        if open:
            import webbrowser

            opened = webbrowser.open(
                f"http://{self.viewer.get_host()}:{self.viewer.get_port()}"
            )
            deadline = time.monotonic() + 10.0
            while opened and len(self.viewer.get_clients()) == 0:
                if time.monotonic() > deadline:
                    break
                time.sleep(0.1)
        if loadModel:
            self.load_viewer_model()

    def load_viewer_model(
        self,
        rootNodeName: str = "robot",
        frame_axis_length: float = 0.2,
        frame_axis_radius: float = 0.01,
        collision_color: Optional[Sequence[float]] = None,
    ) -> None:
        """Load the visual and collision models in the viewer.

        Visuals are displayed after loading, collision shapes and frames are
        hidden. Call :func:`display_collisions` and :func:`display_frames` to
        show them.

        Args:
            rootNodeName: Name of the root node of the robot in the scene.
            frame_axis_length: Length of frame axes in [m].
            frame_axis_radius: Radius of frame axes in [m].
            collision_color: RGBA color applied to all collision shapes. If
                None, each shape keeps its own color.
        """
        self.viewer_root_node_name = rootNodeName
        self.collision_root_node_name = rootNodeName + "/collision"
        self.collision_root_frame = self._scene.add_frame(
            self.collision_root_node_name, show_axes=False
        )
        self.visual_root_node_name = rootNodeName + "/visual"
        self.visual_root_frame = self._scene.add_frame(
            self.visual_root_node_name, show_axes=False
        )
        self.frames_root_node_name = rootNodeName + "/frames"
        self.frames_root_frame = self._scene.add_frame(
            self.frames_root_node_name, show_axes=False
        )
        if self.collision_model is not None:
            for collision in self.collision_model.geometry_objects:
                self.load_viewer_geometry_object(
                    collision, self.collision_root_node_name, collision_color
                )
        self.display_collisions(False)
        if self.visual_model is not None:
            for visual in self.visual_model.geometry_objects:
                self.load_viewer_geometry_object(
                    visual, self.visual_root_node_name
                )
        self.display_visuals(True)
        for frame in self.model.frames:
            frame_name = self.frames_root_node_name + "/" + frame.name
            self.frames[frame_name] = self._scene.add_frame(
                frame_name,
                show_axes=True,
                axes_length=frame_axis_length,
                axes_radius=frame_axis_radius,
            )
        self.display_frames(False)

    def load_viewer_geometry_object(
        self,
        geometry_object: GeometryObject,
        prefix: str,
        color: Optional[Sequence[float]] = None,
    ) -> None:
        """Add one geometry object to the Viser scene.

        Subclasses can override this function to customize how geometries
        are displayed, for instance by changing the arguments passed to the
        scene.

        Args:
            geometry_object: Geometry object to add.
            prefix: Name of the parent node in the scene.
            color: RGBA color of the shape. If None, the color of the
                geometry object is used.

        Raises:
            ValueError: If the shape of the geometry object is not supported.
        """
        import trimesh

        name = prefix + "/" + geometry_object.name
        rgba = (
            geometry_object.mesh_color
            if color is None
            else np.asarray(color, dtype=np.float64)
        )
        if geometry_object.shape == "box":
            handle = self._scene.add_box(
                name,
                dimensions=geometry_object.size,
                color=rgba[:3],
                opacity=rgba[3],
            )
        elif geometry_object.shape == "sphere":
            handle = self._scene.add_icosphere(
                name,
                radius=geometry_object.size[0],
                color=rgba[:3],
                opacity=rgba[3],
            )
        elif geometry_object.shape == "cylinder":
            mesh = trimesh.creation.cylinder(
                radius=geometry_object.size[0],
                height=geometry_object.size[1],
            )
            handle = self._scene.add_mesh_simple(
                name,
                mesh.vertices,
                mesh.faces,
                color=rgba[:3],
                opacity=rgba[3],
            )
        elif geometry_object.shape == "mesh":
            path = geometry_object.mesh_path
            if path.lower().endswith(".dae"):
                mesh = trimesh.load_scene(path)
                mesh.apply_scale(geometry_object.mesh_scale)
                handle = self._scene.add_mesh_trimesh(name, mesh)
            else:
                mesh = trimesh.load_mesh(path)
                mesh.apply_scale(geometry_object.mesh_scale)
                handle = self._scene.add_mesh_simple(
                    name,
                    mesh.vertices,
                    mesh.faces,
                    color=rgba[:3],
                    opacity=rgba[3],
                )
        else:  # pragma: no cover
            raise ValueError(
                f"unsupported geometry shape: {geometry_object.shape}"
            )
        self.frames[name] = handle

    def display(self, q: Optional[np.ndarray] = None) -> None:
        """Display the robot at a given configuration.

        Args:
            q: Configuration vector. If None, the placements already in the
                visualizer's data are displayed as is.
        """
        if q is not None:
            kin.forward_kinematics(self.model, self.data, q)
        if self.collision_root_frame.visible:
            self.update_placements(GeometryType.COLLISION)
        if self.visual_root_frame.visible:
            self.update_placements(GeometryType.VISUAL)
        if self.frames_root_frame.visible:
            self.update_frames()

    def display_collisions(self, visibility: bool) -> None:
        """Toggle display of the collision model.

        Args:
            visibility: If true, display the collision model.
        """
        self.collision_root_frame.visible = visibility
        self.update_placements(GeometryType.COLLISION)

    def display_visuals(self, visibility: bool) -> None:
        """Toggle display of the visual model.

        Args:
            visibility: If true, display the visual model.
        """
        self.visual_root_frame.visible = visibility
        self.update_placements(GeometryType.VISUAL)

    def display_frames(self, visibility: bool) -> None:
        """Toggle display of the model frames.

        Args:
            visibility: If true, display the frames of the model.
        """
        self.frames_root_frame.visible = visibility
        if visibility:
            self.update_frames()

    def update_placements(
        self, geometry_type: GeometryType = GeometryType.VISUAL
    ) -> None:
        """Update geometry placements in the viewer.

        Args:
            geometry_type: Kind of geometries to update, visuals by default.
        """
        if geometry_type == GeometryType.VISUAL:
            geometry_model = self.visual_model
            prefix = self.visual_root_node_name
        else:
            geometry_model = self.collision_model
            prefix = self.collision_root_node_name
        if geometry_model is None:
            return
        for geometry_object in geometry_model.geometry_objects:
            oMi = self.data.oMi[geometry_object.parent_joint]
            oMg = oMi * geometry_object.placement
            handle = self.frames[prefix + "/" + geometry_object.name]
            handle.position = oMg.translation
            handle.wxyz = quaternion_wxyz(oMg.rotation)

    def update_frames(self) -> None:
        """Update model frame placements in the viewer."""
        kin.update_frame_placements(self.model, self.data)
        for frame_id, frame in enumerate(self.model.frames):
            M = self.data.oMf[frame_id]
            handle = self.frames[self.frames_root_node_name + "/" + frame.name]
            handle.position = M.translation
            handle.wxyz = quaternion_wxyz(M.rotation)


def start_viser_visualizer(
    robot: kin.RobotWrapper,
    open: bool = True,  # pylint: disable=redefined-builtin
) -> ViserVisualizer:
    """Open a Viser visualizer in a Web browser.

    Args:
        robot: Robot wrapper with its model and data.
        open: If set (default), open Viser in a new Web browser tab.

    Returns:
        Viser visualizer.
    """
    visualizer = ViserVisualizer(
        robot.model,
        visual_model=robot.visual_model,
        collision_model=robot.collision_model,
    )
    visualizer.init_viewer(open=open)
    visualizer.load_viewer_model()
    return visualizer
