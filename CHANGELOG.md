# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- examples: RB-Y1 carrying a tray along `SE2` waypoints
- kinematics: `build_geom_from_mjcf` to read visual geoms from MJCF files
- kinematics: `build_model_from_mjcf` and `build_model_from_mjcf_xml` MJCF parsers
- kinematics: `RobotWrapper.BuildFromMJCF`
- loaders: `load_robot_description` loads descriptions that only have an MJCF
- loaders: `load_robot_mjcf` to load robot models from MJCF files
- kinematics: `SE2` class of planar rigid transforms
- kinematics: `exp_se2`, `log_se2` and `Jlog_se2` maps on SE(2)
- kinematics: `exp_so2` and `log_so2` maps on SO(2)
- kinematics: `rotation_from_axis_angle` rotation about an axis by an angle
- kinematics: `rotation_from_z_axis` minimal rotation from the z-axis to a unit vector

### Changed

- Bump minimum supported Python version to 3.11

### Removed

- Python 3.10 support (end of life)

## [1.1.0] - 2026-10-19

### Added

- examples: Panda comparing inverse kinematics with and without `ManipulabilityTask`

### Fixed

- kinematics: `center_of_mass` returns a copy rather than a reference
- kinematics: `compute_joint_jacobians` return copies rather than a reference
- kinematics: `jacobian_center_of_mass` returns a copy rather than a reference

## [1.0.0] - 2026-09-27

### Added

- `pinker.kinematics`: C-extension kinematics backend implementing the subset of Pinocchio that Pink used, cross-validated against Pink
- `load_robot_description` and `load_robot_urdf` functions to load a robot from a robot description or a URDF file
- Configuration: `copy` function, where copies share the model, default limits and tangent space
- Configuration: `default_limits` attribute and constructor argument, holding the limits `solve_ik` enforces by default

### Changed

- **Breaking:** Import the library as `pinker`, with its visualizer in `pinker.visualizer`
- **Breaking:** Kinematics types come from `pinker.kinematics` and follow Python's PEP 8 naming conventions, e.g. `model.get_frame_id` instead of `getFrameId`
- **Breaking:** Model getters now raise when a name is not found, where previously in Pink this would have returned a sentinel index
- **Breaking:** Default limits are now in configurations, for instance append to `configuration.default_limits` where you used to set `configuration.model.floating_base_velocity_limit`
- **Breaking:** `Configuration.integrate` returns a new `Configuration` rather than a configuration vector
- Visualize robots with [Viser](https://viser.studio), which serves the scene to a browser, rather than with MeshCat
- Installing compiles the C extension, which requires a C compiler and the NumPy headers
- docs: Move documentation from `doc/` to `docs/`, in the Material theme and the colors of the project
- examples: Port all examples to Viser and rename them to `<robot_description>_<task>.py`

### Removed

- **Breaking:** Collision support, including `SelfCollisionBarrier` and `utils.process_collision_pairs`: stay with Pink if you need collision avoidance
- **Breaking:** Dependency on Pinocchio, replaced by `pinker.kinematics`
- **Breaking:** MeshCat visualization, in particular `start_meshcat_visualizer`
- **Breaking:** utils: `custom_configuration_vector`, `get_root_joint_dim` and `get_joint_idx`, which moved to the kinematics backend
- Dependency on loop-rate-limiters
- examples: Remove the flying dual-arm UR3
- examples: Remove the Panda manipulability comparison
- examples: Remove the two iiwa and yumi collision-avoidance examples
- examples: Remove the yourdfpy visualization

## [0.1.0] - 2026-09-09

Starting this changelog as of Pink 4.4.0.

[unreleased]: https://github.com/pink-kinematics/pinker/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/pink-kinematics/pinker/releases/tag/v1.1.0
[1.0.0]: https://github.com/pink-kinematics/pinker/releases/tag/v1.0.0
[0.1.0]: https://github.com/pink-kinematics/pinker/releases/tag/v0.1.0
