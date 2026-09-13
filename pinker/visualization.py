# SPDX-License-Identifier: Apache-2.0

"""Visualization helpers."""

from . import kinematics as kin
from .kinematics.visualize import ViserVisualizer


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
    robot.set_visualizer(visualizer, init=False)
    visualizer.init_viewer(open=open)
    visualizer.load_viewer_model()
    return visualizer
