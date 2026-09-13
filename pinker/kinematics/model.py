# SPDX-License-Identifier: Apache-2.0

"""Robot model and data.

A :class:`Model` describes the kinematic tree of a robot: joints, frames,
inertias and limits. A :class:`Data` holds the buffers written by the
kinematics algorithms (joint and frame placements, Jacobians).
"""

from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np

from .joints import JointModel, joint_has_configuration_limit
from .se3 import SE3, Inertia


class ReferenceFrame(IntEnum):
    """Reference frame for velocities and Jacobians."""

    WORLD = 0
    LOCAL = 1
    LOCAL_WORLD_ALIGNED = 2


class FrameType(IntEnum):
    """Frame types, with the same values."""

    OP_FRAME = 1
    JOINT = 2
    FIXED_JOINT = 4
    BODY = 8
    SENSOR = 16


class Frame:
    """Coordinate frame attached to a joint of the kinematic tree.

    Attributes:
        name: Name of the frame.
        parent_joint: Index of the joint supporting the frame.
        parent_frame: Index of the previous frame in the tree.
        placement: Pose of the frame in the support joint frame.
        type: Frame type.
    """

    def __init__(
        self,
        name: str,
        parent_joint: int,
        parent_frame: int,
        placement: SE3,
        type: FrameType,
    ):
        """Initialize frame.

        Args:
            name: Name of the frame.
            parent_joint: Index of the joint supporting the frame.
            parent_frame: Index of the previous frame in the tree.
            placement: Pose of the frame in the support joint frame.
            type: Frame type.
        """
        self.name = name
        self.parent_joint = parent_joint
        self.parent_frame = parent_frame
        self.placement = placement
        self.type = type

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Human-readable representation of the frame.
        """
        return f"Frame({self.name!r}, joint={self.parent_joint})"


class Model:
    """Kinematic model of a robot.

    Joints are indexed by joint id, the first one being the universe; limits
    are indexed by configuration or tangent coordinate.

    Attributes:
        effort_limit: Maximum effort of each tangent coordinate.
        frames: Frames of the model, indexed by frame id.
        inertias: Inertia of the body attached to each joint, in the frame of
            that joint.
        joint_placements: Placement of each joint in the frame of its parent
            joint.
        joints: Joint models of the kinematic tree.
        lower_position_limit: Lower bound of each configuration coordinate.
        name: Name of the robot model.
        names: Name of each joint.
        nq: Dimension of the configuration vector.
        nv: Dimension of the tangent vector.
        parents: Index of the parent joint of each joint.
        supports: Indexes of the joints supporting each joint, from the
            universe down to that joint.
        upper_position_limit: Upper bound of each configuration coordinate.
        velocity_limit: Maximum velocity of each tangent coordinate.
    """

    effort_limit: np.ndarray
    frames: List[Frame]
    inertias: List[Inertia]
    joint_placements: List[SE3]
    joints: List[JointModel]
    lower_position_limit: np.ndarray
    name: str
    names: List[str]
    nq: int
    nv: int
    parents: List[int]
    supports: List[List[int]]
    upper_position_limit: np.ndarray
    velocity_limit: np.ndarray

    def __init__(self):
        """Initialize an empty model with only the universe joint."""
        universe = JointModel(0)
        universe.id = 0
        self.name = ""
        self.joints = [universe]
        self.names = ["universe"]
        self.parents = [0]
        self.joint_placements = [SE3.Identity()]
        self.inertias = [Inertia.Zero()]
        self.frames = [
            Frame("universe", 0, 0, SE3.Identity(), FrameType.FIXED_JOINT)
        ]
        self._frame_ids = {"universe": 0}
        self.supports = [[0]]
        self.nq = 0
        self.nv = 0
        self.lower_position_limit = np.zeros(0)
        self.upper_position_limit = np.zeros(0)
        self.velocity_limit = np.zeros(0)
        self.effort_limit = np.zeros(0)
        self._cache = None

    @property
    def njoints(self) -> int:
        """Number of joints, including the universe.

        Returns:
            Number of joints in the model.
        """
        return len(self.joints)

    @property
    def nframes(self) -> int:
        """Number of frames.

        Returns:
            Number of frames in the model.
        """
        return len(self.frames)

    def add_joint(
        self,
        parent_id: int,
        joint_model: JointModel,
        placement: SE3,
        name: str,
        max_effort: Optional[np.ndarray] = None,
        max_velocity: Optional[np.ndarray] = None,
        min_config: Optional[np.ndarray] = None,
        max_config: Optional[np.ndarray] = None,
    ) -> int:
        """Add a joint to the kinematic tree.

        Args:
            parent_id: Index of the parent joint.
            joint_model: Joint model, e.g. from ``JointModelRZ()``.
            placement: Pose of the joint frame in the parent joint frame.
            name: Name of the new joint.
            max_effort: Maximum joint efforts (default: unbounded).
            max_velocity: Maximum joint velocities (default: unbounded).
            min_config: Lower configuration limits (default: unbounded).
            max_config: Upper configuration limits (default: unbounded).

        Returns:
            Index of the new joint.
        """
        joint = JointModel(joint_model.jtype, joint_model.axis)
        joint.id = len(self.joints)
        joint.idx_q = self.nq
        joint.idx_v = self.nv
        self.joints.append(joint)
        self.names.append(name)
        self.parents.append(parent_id)
        self.joint_placements.append(placement.copy())
        self.inertias.append(Inertia.Zero())
        self.supports.append(self.supports[parent_id] + [joint.id])
        self.nq += joint.nq
        self.nv += joint.nv

        # As in Pinocchio, default limits are the largest double, not inf
        dbl_max = np.finfo(np.float64).max
        if min_config is None:
            min_config = np.full(joint.nq, -dbl_max)
        if max_config is None:
            max_config = np.full(joint.nq, dbl_max)
        if max_velocity is None:
            max_velocity = np.full(joint.nv, dbl_max)
        if max_effort is None:
            max_effort = np.full(joint.nv, dbl_max)
        self.lower_position_limit = np.hstack(
            [self.lower_position_limit, min_config]
        )
        self.upper_position_limit = np.hstack(
            [self.upper_position_limit, max_config]
        )
        self.velocity_limit = np.hstack([self.velocity_limit, max_velocity])
        self.effort_limit = np.hstack([self.effort_limit, max_effort])
        self._cache = None
        return joint.id

    def add_joint_frame(self, joint_id: int, previous_frame: int = -1) -> int:
        """Add a JOINT type frame coinciding with a joint of the model.

        Args:
            joint_id: Index of the joint the frame coincides with.
            previous_frame: Index of the previous frame in the tree. A
                negative value denotes the universe frame.

        Returns:
            Index of the new frame.
        """
        if previous_frame < 0:
            previous_frame = 0
        frame = Frame(
            self.names[joint_id],
            joint_id,
            previous_frame,
            SE3.Identity(),
            FrameType.JOINT,
        )
        return self.add_frame(frame)

    def append_body_to_joint(
        self,
        joint_id: int,
        inertia: Inertia,
        placement: Optional[SE3] = None,
    ) -> None:
        """Append body inertia to a joint of the model.

        Args:
            joint_id: Index of the supporting joint.
            inertia: Inertia of the body, in the body frame.
            placement: Pose of the body frame in the joint frame.
        """
        if placement is None:
            placement = SE3.Identity()
        self.inertias[joint_id] = self.inertias[joint_id] + inertia.displaced(
            placement
        )
        self._cache = None

    def add_body_frame(
        self,
        name: str,
        parent_joint: int,
        placement: Optional[SE3] = None,
        previous_frame: int = -1,
    ) -> int:
        """Add a BODY type frame to the model.

        Args:
            name: Name of the new frame.
            parent_joint: Index of the joint supporting the frame.
            placement: Pose of the frame in the support joint frame,
                defaulting to the identity.
            previous_frame: Index of the previous frame in the tree. A
                negative value denotes the universe frame.

        Returns:
            Index of the new frame.
        """
        if placement is None:
            placement = SE3.Identity()
        if previous_frame < 0:
            previous_frame = 0
        frame = Frame(
            name, parent_joint, previous_frame, placement, FrameType.BODY
        )
        return self.add_frame(frame)

    def add_frame(self, frame: Frame) -> int:
        """Add a frame to the model.

        Args:
            frame: Frame to add.

        Returns:
            Index of the new frame.
        """
        frame_id = len(self.frames)
        self.frames.append(frame)
        if frame.name not in self._frame_ids:
            self._frame_ids[frame.name] = frame_id
        self._cache = None
        return frame_id

    def _reindex_frames(self) -> None:
        """Rebuild the frame name index (frames may be renamed in place)."""
        self._frame_ids = {}
        for frame_id, frame in enumerate(self.frames):
            if frame.name not in self._frame_ids:
                self._frame_ids[frame.name] = frame_id

    def _find_frame_id(self, name: str) -> Optional[int]:
        """Index of the frame with a given name, if the model has one.

        Args:
            name: Name of the frame to look up.

        Returns:
            Index of the frame, or None if the model has no such frame.
        """
        frame_id = self._frame_ids.get(name)
        if frame_id is not None and self.frames[frame_id].name == name:
            return frame_id
        # Cache miss, or stale entry after frames were renamed in place
        self._reindex_frames()
        return self._frame_ids.get(name)

    def exist_frame(self, name: str) -> bool:
        """Check whether a frame name exists in the model.

        Args:
            name: Name of the frame to look up.

        Returns:
            True if the model has a frame with this name.
        """
        return self._find_frame_id(name) is not None

    def get_frame_id(self, name: str) -> int:
        """Index of the frame with a given name.

        Args:
            name: Name of the frame to look up.

        Returns:
            Index of the frame.

        Raises:
            ValueError: If the model has no frame with this name. Pinocchio
                returns ``nframes`` in that case; we raise so that all Model
                getters behave the same way.
        """
        frame_id = self._find_frame_id(name)
        if frame_id is None:
            raise ValueError(f"no frame named '{name}' in this model")
        return frame_id

    def exist_joint_name(self, name: str) -> bool:
        """Check whether a joint name exists in the model.

        Args:
            name: Name of the joint to look up.

        Returns:
            True if the model has a joint with this name.
        """
        return name in self.names

    def get_joint_id(self, name: str) -> int:
        """Index of the joint with a given name.

        Args:
            name: Name of the joint to look up.

        Returns:
            Index of the joint.

        Raises:
            ValueError: If the model has no joint with this name. Pinocchio
                returns ``njoints`` in that case; we raise so that all Model
                getters behave the same way.
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise ValueError(
                f"no joint named '{name}' in this model"
            ) from None

    def get_joint_tangent_id(self, name: str) -> int:
        """Index of a joint in the tangent vector.

        Args:
            name: Name of the joint to look up.

        Returns:
            Index of the first tangent coordinate of the joint, so that its
            coordinates are ``v[idx_v:idx_v + joint.nv]``.

        Raises:
            ValueError: If the model has no joint with this name.
        """
        return self.joints[self.get_joint_id(name)].idx_v

    def get_root_joint_dim(self) -> Tuple[int, int]:
        """Count configuration and tangent dimensions of the root joint.

        Returns:
            Pair ``(nq, nv)`` of the configuration and tangent dimensions of
            the root joint, or ``(0, 0)`` if the model has no root joint.
        """
        if self.exist_joint_name("root_joint"):
            root_joint = self.joints[self.get_joint_id("root_joint")]
            return root_joint.nq, root_joint.nv
        return 0, 0

    def has_configuration_limit(self) -> np.ndarray:
        """Boolean array flagging configuration coordinates with limits.

        Returns:
            One boolean per configuration coordinate, True if that coordinate
            has a position limit.
        """
        flags: List[bool] = []
        for joint in self.joints[1:]:
            flags.extend(joint_has_configuration_limit(joint.jtype))
        return np.array(flags, dtype=bool)

    def create_data(self) -> "Data":
        """Create a data buffer matching this model.

        Returns:
            New data for this model.
        """
        return Data(self)

    def _packed(self) -> dict:
        """Contiguous arrays describing the model, for the C kernels.

        The cache is rebuilt whenever the model is modified.

        Returns:
            Dictionary of contiguous arrays, keyed by quantity name.
        """
        if self._cache is not None:
            return self._cache
        nj = self.njoints
        nf = self.nframes
        cache = {
            "jtype": np.array(
                [joint.jtype for joint in self.joints], dtype=np.int32
            ),
            "parent": np.array(self.parents, dtype=np.int32),
            "idx_q": np.array(
                [max(joint.idx_q, 0) for joint in self.joints], dtype=np.int32
            ),
            "idx_v": np.array(
                [max(joint.idx_v, 0) for joint in self.joints], dtype=np.int32
            ),
            "axis": np.zeros((nj, 3)),
            "jp_rot": np.zeros((nj, 3, 3)),
            "jp_trans": np.zeros((nj, 3)),
            "mass": np.array([inertia.mass for inertia in self.inertias]),
            "lever": np.ascontiguousarray(
                np.vstack([inertia.lever for inertia in self.inertias])
            ),
            "fparent": np.array(
                [frame.parent_joint for frame in self.frames], dtype=np.int32
            ),
            "fp_rot": np.zeros((nf, 3, 3)),
            "fp_trans": np.zeros((nf, 3)),
        }
        for j, joint in enumerate(self.joints):
            if joint.axis is not None:
                cache["axis"][j] = joint.axis
            cache["jp_rot"][j] = self.joint_placements[j].rotation
            cache["jp_trans"][j] = self.joint_placements[j].translation
        for f, frame in enumerate(self.frames):
            cache["fp_rot"][f] = frame.placement.rotation
            cache["fp_trans"][f] = frame.placement.translation
        # Tangent-space columns of each joint's supporting kinematic chain
        support_cols: List[np.ndarray] = []
        for j in range(nj):
            cols: List[int] = []
            for ancestor in self.supports[j]:
                joint = self.joints[ancestor]
                cols.extend(range(joint.idx_v, joint.idx_v + joint.nv))
            support_cols.append(np.array(cols, dtype=np.int32))
        cache["support_cols"] = support_cols
        self._cache = cache
        return cache

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Human-readable representation of the model.
        """
        return (
            f"Model({self.name!r}, nq={self.nq}, nv={self.nv}, "
            f"njoints={self.njoints})"
        )


class _SE3View:
    """List-like access to placements stored in packed arrays."""

    def __init__(self, rotations: np.ndarray, translations: np.ndarray):
        """Wrap (n, 3, 3) rotations and (n, 3) translations.

        Args:
            rotations: Rotations of the placements, of shape (n, 3, 3).
            translations: Translations of the placements, of shape (n, 3).
        """
        self._rotations = rotations
        self._translations = translations

    def __getitem__(self, index: int) -> SE3:
        """Placement at a given index, sharing memory with the buffers.

        Args:
            index: Index of the placement.

        Returns:
            Placement at this index, sharing memory with the buffers.

        Raises:
            IndexError: If the index is out of range.
        """
        if index >= self._rotations.shape[0]:
            raise IndexError(f"index {index} out of range")
        return SE3._new(self._rotations[index], self._translations[index])

    def __len__(self) -> int:
        """Number of placements.

        Returns:
            Number of placements in the view.
        """
        return self._rotations.shape[0]


class Data:
    """Buffers written by the kinematics algorithms.

    Attributes:
        J: Full model Jacobian (6 x nv), world frame, filled by
            compute_joint_jacobians.
        Jcom: Center-of-mass Jacobian (3 x nv).
        com: List whose first element is the robot's center of mass.
        mass: List whose first element is the robot's total mass.
    """

    def __init__(self, model: Optional[Model] = None):
        """Allocate buffers for a model.

        Args:
            model: Model to allocate buffers for. If None, the data is an
                empty shell to be filled by :func:`copy`.
        """
        if model is None:
            return  # empty shell, filled by copy()
        nj, nf, nv = model.njoints, model.nframes, model.nv
        self._oMi_rot = np.tile(np.eye(3), (nj, 1, 1))
        self._oMi_trans = np.zeros((nj, 3))
        self._oMf_rot = np.tile(np.eye(3), (nf, 1, 1))
        self._oMf_trans = np.zeros((nf, 3))
        self.J = np.zeros((6, nv))
        self.Jcom = np.zeros((3, nv))
        self.com = [np.zeros(3)]
        self.mass = [0.0]

    @property
    def oMi(self) -> _SE3View:
        """Joint placements in the world frame.

        Returns:
            List-like view of the joint placements.
        """
        return _SE3View(self._oMi_rot, self._oMi_trans)

    @property
    def oMf(self) -> _SE3View:
        """Frame placements in the world frame.

        Returns:
            List-like view of the frame placements.
        """
        return _SE3View(self._oMf_rot, self._oMf_trans)

    def copy(self) -> "Data":
        """Copy of this data with its own buffers.

        Returns:
            New data with the same values in freshly allocated buffers.
        """
        other = Data()
        other._oMi_rot = self._oMi_rot.copy()
        other._oMi_trans = self._oMi_trans.copy()
        other._oMf_rot = self._oMf_rot.copy()
        other._oMf_trans = self._oMf_trans.copy()
        other.J = self.J.copy()
        other.Jcom = self.Jcom.copy()
        other.com = [com.copy() for com in self.com]
        other.mass = list(self.mass)
        return other
