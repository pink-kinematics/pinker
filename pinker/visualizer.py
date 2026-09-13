# SPDX-License-Identifier: Apache-2.0

"""Robot visualization with Viser."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from . import kinematics as kin
from .kinematics import GeometryModel, GeometryObject
from .kinematics.model import Data, Model
from .kinematics.so3 import quaternion_wxyz

if TYPE_CHECKING:  # viser is imported lazily, when starting the viewer
    import viser


class ViserVisualizer:
    """A robot visualizer using Viser, like Pinocchio's.

    The viewer and the scene nodes are created when the visualizer starts:
    :func:`init_viewer` connects to a Viser server, then
    :func:`load_viewer_model` adds the robot's visuals and frames to its
    scene.

    Attributes:
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
    ):
        """Initialize visualizer from a model and its geometry models.

        Args:
            model: Robot model to visualize.
            visual_model: Geometry model with the robot's visuals.
        """
        self.model = model
        self.visual_model = visual_model
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
    ) -> None:
        """Load the visual model in the viewer.

        Args:
            rootNodeName: Name of the root node of the robot in the scene.
            frame_axis_length: Length of frame axes in [m].
            frame_axis_radius: Radius of frame axes in [m].
        """
        self.viewer_root_node_name = rootNodeName
        self.visual_root_node_name = rootNodeName + "/visual"
        self.visual_root_frame = self._scene.add_frame(
            self.visual_root_node_name, show_axes=False
        )
        self.frames_root_node_name = rootNodeName + "/frames"
        self.frames_root_frame = self._scene.add_frame(
            self.frames_root_node_name, show_axes=False
        )
        if self.visual_model is not None:
            for visual in self.visual_model.geometry_objects:
                self._load_geometry_object(visual, self.visual_root_node_name)
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

    def _load_geometry_object(
        self, geometry_object: GeometryObject, prefix: str
    ) -> None:
        """Add one geometry object to the Viser scene.

        Args:
            geometry_object: Geometry object to add.
            prefix: Name of the parent node in the scene.

        Raises:
            ValueError: If the shape of the geometry object is not supported.
        """
        import trimesh

        name = prefix + "/" + geometry_object.name
        color = geometry_object.mesh_color
        if geometry_object.shape == "box":
            handle = self._scene.add_box(
                name,
                dimensions=geometry_object.size,
                color=color[:3],
                opacity=color[3],
            )
        elif geometry_object.shape == "sphere":
            handle = self._scene.add_icosphere(
                name,
                radius=geometry_object.size[0],
                color=color[:3],
                opacity=color[3],
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
                color=color[:3],
                opacity=color[3],
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
                    color=color[:3],
                    opacity=color[3],
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
        if self.visual_root_frame.visible:
            self.update_placements()
        if self.frames_root_frame.visible:
            self.update_frames()

    def display_visuals(self, visibility: bool) -> None:
        """Toggle display of the visual model.

        Args:
            visibility: If true, display the visual model.
        """
        self.visual_root_frame.visible = visibility
        self.update_placements()

    def display_frames(self, visibility: bool) -> None:
        """Toggle display of the model frames.

        Args:
            visibility: If true, display the frames of the model.
        """
        self.frames_root_frame.visible = visibility
        if visibility:
            self.update_frames()

    def update_placements(self) -> None:
        """Update visual geometry placements in the viewer."""
        if self.visual_model is None:
            return
        for geometry_object in self.visual_model.geometry_objects:
            oMi = self.data.oMi[geometry_object.parent_joint]
            oMg = oMi * geometry_object.placement
            handle = self.frames[
                self.visual_root_node_name + "/" + geometry_object.name
            ]
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
    visualizer = ViserVisualizer(robot.model, visual_model=robot.visual_model)
    visualizer.init_viewer(open=open)
    visualizer.load_viewer_model()
    return visualizer
