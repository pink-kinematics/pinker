# SPDX-License-Identifier: Apache-2.0

"""Tests for the Viser visualizer."""

import os
import sys
import types
import unittest

import numpy as np

from pinker import kinematics as kin
from pinker import load_robot_urdf
from pinker.visualizer import ViserVisualizer

URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "geometries.urdf",
)


class FakeHandle:
    """Scene node handle recording what the visualizer sets on it."""

    def __init__(self, kind: str, kwargs: dict):
        """Record the creation call of the node."""
        self.kind = kind
        self.kwargs = kwargs
        self.visible = True
        self.position = np.zeros(3)
        self.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self.removed = False

    def remove(self) -> None:
        """Remove node from the scene."""
        self.removed = True


class FakeScene:
    """Viser scene recording the nodes added to it."""

    def __init__(self):
        """Initialize an empty scene."""
        self.nodes = {}

    def _add(self, kind: str, name: str, **kwargs) -> FakeHandle:
        handle = FakeHandle(kind, kwargs)
        self.nodes[name] = handle
        return handle

    def add_frame(self, name, **kwargs):
        """Add a coordinate frame."""
        return self._add("frame", name, **kwargs)

    def add_box(self, name, **kwargs):
        """Add a box."""
        return self._add("box", name, **kwargs)

    def add_icosphere(self, name, **kwargs):
        """Add a sphere."""
        return self._add("icosphere", name, **kwargs)

    def add_mesh_simple(self, name, vertices, faces, **kwargs):
        """Add a mesh."""
        return self._add("mesh", name, **kwargs)


class FakeViewer:
    """Viser server with a fake scene."""

    def __init__(self):
        """Initialize viewer."""
        self.scene = FakeScene()


def fake_trimesh() -> types.ModuleType:
    """Minimal trimesh module, for environments where it is not installed."""
    trimesh = types.ModuleType("trimesh")
    creation = types.ModuleType("trimesh.creation")

    def cylinder(radius, height):
        return types.SimpleNamespace(
            vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]])
        )

    creation.cylinder = cylinder
    trimesh.creation = creation
    return trimesh


class TestViserVisualizer(unittest.TestCase):
    """Test the Viser visualizer with visual and collision models."""

    @classmethod
    def setUpClass(cls):
        """Use a minimal trimesh if the actual package is not installed."""
        cls.saved_trimesh = sys.modules.get("trimesh")
        try:
            import trimesh  # noqa: F401
        except ImportError:
            sys.modules["trimesh"] = fake_trimesh()

    @classmethod
    def tearDownClass(cls):
        """Restore the trimesh module."""
        if cls.saved_trimesh is None:
            sys.modules.pop("trimesh", None)
        else:
            sys.modules["trimesh"] = cls.saved_trimesh

    def setUp(self):
        """Load the robot in a visualizer with a fake viewer."""
        self.robot = load_robot_urdf(URDF_PATH)
        self.viz = ViserVisualizer(
            self.robot.model,
            visual_model=self.robot.visual_model,
            collision_model=self.robot.collision_model,
        )
        self.viz.viewer = FakeViewer()  # init_viewer would import viser
        self.scene = self.viz.viewer.scene

    def test_collision_nodes(self):
        """Collision shapes have their own nodes, hidden by default."""
        self.viz.load_viewer_model()
        collision_nodes = sorted(
            name
            for name in self.viz.frames
            if name.startswith("robot/collision/")
        )
        self.assertEqual(
            collision_nodes,
            [
                "robot/collision/arm_0",
                "robot/collision/base_0",
                "robot/collision/base_1",
                "robot/collision/tool_0",
            ],
        )
        self.assertFalse(self.viz.collision_root_frame.visible)
        self.assertTrue(self.viz.visual_root_frame.visible)

    def test_collision_color(self):
        """A collision color applies to collision shapes only."""
        self.viz.load_viewer_model(collision_color=[1.0, 0.0, 0.0, 0.5])
        collision_box = self.scene.nodes["robot/collision/base_0"]
        self.assertTrue(np.allclose(collision_box.kwargs["color"], [1, 0, 0]))
        self.assertEqual(collision_box.kwargs["opacity"], 0.5)
        visual_box = self.scene.nodes["robot/visual/base_0"]
        self.assertTrue(
            np.allclose(visual_box.kwargs["color"], [0.1, 0.2, 0.3])
        )

    def test_display_collisions(self):
        """Collision shapes follow the configuration once displayed."""
        self.viz.load_viewer_model()
        self.viz.display_collisions(True)
        self.assertTrue(self.viz.collision_root_frame.visible)
        q = self.robot.q0.copy()
        q[0] = 0.7
        self.viz.display(q)
        kin.frames_forward_kinematics(self.robot.model, self.robot.data, q)
        arm_sphere = self.robot.collision_model.geometry_objects[2]
        oMg = (
            self.robot.data.oMf[arm_sphere.parent_frame]
            * self.robot.model.frames[
                arm_sphere.parent_frame
            ].placement.inverse()
            * arm_sphere.placement
        )
        handle = self.viz.frames["robot/collision/" + arm_sphere.name]
        self.assertTrue(np.allclose(handle.position, oMg.translation))
        self.assertTrue(
            np.allclose(handle.wxyz, kin.quaternion_wxyz(oMg.rotation))
        )

    def test_load_viewer_geometry_object_override(self):
        """Subclasses can customize how each geometry is added."""
        loaded = []

        class Visualizer(ViserVisualizer):
            def load_viewer_geometry_object(
                self, geometry_object, prefix, color=None
            ):
                loaded.append(prefix + "/" + geometry_object.name)
                super().load_viewer_geometry_object(
                    geometry_object, prefix, color
                )

        viz = Visualizer(
            self.robot.model,
            visual_model=self.robot.visual_model,
            collision_model=self.robot.collision_model,
        )
        viz.viewer = FakeViewer()
        viz.load_viewer_model()
        self.assertEqual(
            len(loaded),
            self.robot.visual_model.ngeoms + self.robot.collision_model.ngeoms,
        )


if __name__ == "__main__":
    unittest.main()
