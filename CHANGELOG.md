# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[unreleased]: https://github.com/pink-kinematics/pinker/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/pink-kinematics/pinker/releases/tag/v1.0.0
[0.1.0]: https://github.com/pink-kinematics/pinker/releases/tag/v0.1.0
