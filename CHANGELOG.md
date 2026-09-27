# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `pinker.kinematics`: kinematics backend, written as a single C extension
  with a thin Python layer, implementing the subset of Pinocchio that Pink
  used: URDF parsing, forward kinematics, frame and joint Jacobians,
  Lie-group operations on SE(3) and on configuration spaces, center of mass
  and its Jacobian. Its outputs are cross-validated against Pinocchio
- `load_robot_description` and `load_robot_urdf`, loading a robot from a robot
  description or from a URDF file
- Configuration: `default_limits` attribute and constructor argument, holding
  the limits `solve_ik` enforces unless it is given its own
- Configuration: `copy` function, sharing the model, default limits and
  tangent space of the configuration it copies

### Changed

- **Breaking:** The library is imported as `pinker`, and its visualizer lives
  in `pinker.visualizer`
- **Breaking:** Kinematics types come from `pinker.kinematics` rather than
  from Pinocchio, and follow Python naming: `model.get_frame_id` and
  `model.lower_position_limit` instead of `getFrameId` and
  `lowerPositionLimit`, `SE3.toarray` and `Motion.asarray` instead of the `np`
  property. Model getters raise when a name is not found, where Pinocchio
  returns a sentinel index
- **Breaking:** Default limits are stored on the configuration rather than
  cached on the robot model, which Pinker no longer adds attributes to: append
  to `configuration.default_limits` where you used to set
  `configuration.model.floating_base_velocity_limit`
- **Breaking:** `Configuration.integrate` returns a new `Configuration` rather
  than a configuration vector
- Robots are visualized with [Viser](https://viser.studio), which serves the
  scene to a browser, rather than with MeshCat
- Installing compiles the C extension, which requires a C compiler and the
  NumPy headers
- examples: Ported to Viser, and named `<robot_description>_<task>.py` after
  the robot description they load and what they do with it
- docs: Move the documentation from `doc/` to `docs/`, in the Material theme
  and the colors of the project

### Removed

- **Breaking:** Dependency on Pinocchio, replaced by `pinker.kinematics`
- **Breaking:** Collision support, which `pinker.kinematics` does not
  implement: `SelfCollisionBarrier`, the collision model and data of
  `Configuration` and `RobotWrapper`, and `utils.process_collision_pairs`.
  Stay with Pink if you need collision avoidance
- **Breaking:** MeshCat visualization, in particular
  `start_meshcat_visualizer`
- **Breaking:** utils: `custom_configuration_vector`, `get_root_joint_dim` and
  `get_joint_idx`, which moved to the kinematics backend as
  `kinematics.custom_configuration`, `Model.get_root_joint_dim` and
  `Model.get_joint_tangent_id`
- Dependency on loop-rate-limiters
- examples: Remove the two iiwa and yumi collision-avoidance examples
- examples: Remove the flying dual-arm UR3
- examples: Remove the Panda manipulability comparison
- examples: Remove the yourdfpy visualization

## [0.1.0] - 2026-09-09

Starting this changelog as of Pink 4.4.0.

[unreleased]: https://github.com/pink-kinematics/pink/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pink-kinematics/pinker/releases/tag/v0.1.0
