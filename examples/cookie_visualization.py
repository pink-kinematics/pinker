# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["pinker", "pycollada", "robot_descriptions >=3.1.0",
# "trimesh", "viser"]
#
# [tool.uv.sources]
# pinker = { path = "..", editable = true }
# ///

"""Load a robot description and visualize it in Viser."""

import time

import pinker
from pinker.visualizer import start_viser_visualizer

if __name__ == "__main__":
    robot = pinker.load_robot_description(
        "cookie_description", root_joint="free_flyer"
    )
    viz = start_viser_visualizer(robot)
    viz.display_frames(True)
    viz.display(robot.q0)
    print("Robot displayed, press Ctrl-C to quit.")
    while True:
        time.sleep(1.0)
