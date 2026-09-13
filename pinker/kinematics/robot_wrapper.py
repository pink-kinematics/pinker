# SPDX-License-Identifier: Apache-2.0

"""Convenience wrapper bundling a model with its data."""

from typing import List, Optional

import numpy as np

from .algorithms import neutral
from .geometry import GeometryModel
from .joints import JointModel
from .model import Data, Model
from .urdf import build_geom_from_urdf, build_model_from_urdf


class RobotWrapper:
    """Robot model with its data and neutral configuration.

    Attributes:
        model: Robot model.
        data: Data buffers for the model.
        q0: Neutral configuration.
        visual_model: Geometry model with the robot's visuals, when built
            from URDF.
    """

    model: Model
    data: Data
    q0: np.ndarray
    visual_model: Optional[GeometryModel]

    def __init__(
        self,
        model: Model,
        visual_model: Optional[GeometryModel] = None,
    ):
        """Wrap a robot model.

        Args:
            model: Robot model to wrap.
            visual_model: Geometry model with the robot's visuals.
        """
        self.model = model
        self.data = model.create_data()
        self.q0 = neutral(model)
        self.visual_model = visual_model

    @staticmethod
    def BuildFromURDF(
        filename: str,
        package_dirs: Optional[List[str]] = None,
        root_joint: Optional[JointModel] = None,
    ) -> "RobotWrapper":
        """Build a robot wrapper from a URDF file.

        Args:
            filename: Path to the URDF file.
            package_dirs: Directories where mesh files are looked up.
            root_joint: Optional joint connecting the root link to the world.

        Returns:
            Robot wrapper.
        """
        if isinstance(package_dirs, str):
            package_dirs = [package_dirs]
        model = build_model_from_urdf(filename, root_joint)
        visual_model = build_geom_from_urdf(model, filename, package_dirs)
        return RobotWrapper(model, visual_model)

    @property
    def nq(self) -> int:
        """Number of configuration variables.

        Returns:
            Dimension of the configuration vector of the model.
        """
        return self.model.nq

    @property
    def nv(self) -> int:
        """Number of tangent-space variables.

        Returns:
            Dimension of the tangent vector of the model.
        """
        return self.model.nv
