# Pinker

[![Build](https://img.shields.io/github/actions/workflow/status/pink-kinematics/pinker/ci.yml?branch=main)](https://github.com/pink-kinematics/pinker/actions)
[![Documentation](https://img.shields.io/github/actions/workflow/status/pink-kinematics/pinker/docs.yml?branch=main&label=docs)](https://pink-kinematics.github.io/pinker/)
[![Coverage](https://coveralls.io/repos/github/pink-kinematics/pinker/badge.svg?branch=main)](https://coveralls.io/github/pink-kinematics/pinker?branch=main)
[![PyPI version](https://img.shields.io/pypi/v/pinker)](https://pypi.org/project/pinker/)

**P**ython **in**verse **k**inematics for **e**mbedded **r**obots.

Pinker is a leaner version of [Pink](https://github.com/pink-kinematics/pink/) aimed in particular at single-board computers. It ships its own kinematics backend, so the whole library is NumPy, a QP solver and one C file that takes seconds to compile. Pinker is API-compatible with Pink 4.4.0 and produces the same results at the same speed ([checks](#benchmark)).

## Installation

You can install the library from PyPI:

```console
pip install pinker
```

## Usage

Pinker solves differential inverse kinematics by [weighted tasks](https://scaron.info/robot-locomotion/inverse-kinematics.html). A task is defined by a *residual* function $e(q)$ of the robot configuration $q \in \mathcal{C}$ to be driven to zero. For instance, putting a foot position $p_{foot}(q)$ at a given target $p_{foot}^{\star}$ can be described by the position residual:

$$
e(q) = p_{foot}^{\star} - p_{foot}(q)
$$

In differential inverse kinematics, we compute a velocity $v \in \mathfrak{c}$ that satisfies the first-order differential equation:

$$
J_e(q) v = \dot{e}(q) = -\alpha e(q)
$$

where $J\_e(q) := \frac{\partial e}{\partial q}$ is the [task Jacobian](https://scaron.info/robotics/jacobian-of-a-kinematic-task-and-derivatives-on-manifolds.html). We can define multiple tasks, but some of them will come into conflict if they can't be all fully achieved at the same time. Conflicts are resolved by casting all objectives to a common unit, and weighing these normalized objectives relative to each other. We also include configuration and velocity limits, making our overall optimization problem a quadratic program:

$$
\begin{align}
\underset{v \in \mathfrak{c}}{\text{minimize}} \ & \sum_{\text{task } e} \Vert J_e(q) v + \alpha e(q) \Vert^2_{W_e} \\
\text{subject to} \ & v_{\text{min}}(q) \leq v \leq v_{\text{max}}(q)
\end{align}
$$

Pinker provides an API to describe the problem as tasks with targets, and automatically build and solve the underlying quadratic program.

### Task costs

Here is the example of a biped robot that controls the position and orientation of its base, left and right contact frames. A fourth "posture" task, giving a preferred angle for each joint, is added for regularization:

```python
from pinker.tasks import FrameTask, PostureTask

tasks = {
    "base": FrameTask(
        "base",
        position_cost=1.0,              # [cost] / [m]
        orientation_cost=1.0,           # [cost] / [rad]
    ),
    "left_contact": FrameTask(
        "left_contact",
        position_cost=[0.1, 0.0, 0.1],  # [cost] / [m]
        orientation_cost=0.0,           # [cost] / [rad]
    ),
    "right_contact": FrameTask(
        "right_contact",
        position_cost=[0.1, 0.0, 0.1],  # [cost] / [m]
        orientation_cost=0.0,           # [cost] / [rad]
    ),
    "posture": PostureTask(
        cost=1e-3,                      # [cost] / [rad]
    ),
}
```

Orientation (similarly position) costs can be scalars or 3D vectors. They specify how much each radian of angular error "costs" in the overall normalized objective. When using 3D vectors, components are weighted anisotropically along each axis of the body frame.

### Task targets

Aside from their costs, most tasks take a second set of parameters called *target*. For example, a frame task aims for a target transform, while a posture task aims for a target configuration vector. Targets are set by the `set_target` function:

```python
    tasks["posture"].set_target(
        [1.0, 0.0, 0.0, 0.0] +           # floating base quaternion
        [0.0, 0.0, 0.0] +                # floating base position
        [0.0, 0.2, 0.0, 0.0, -0.2, 0.0]  # joint angles
    )
```

Body tasks can be initialized, for example, from the robot's neutral configuration:

```python
from pinker import Configuration, load_robot_description, solve_ik

robot = load_robot_description("ur3_official_description")
configuration = Configuration(robot.model, robot.data, robot.q0)
for body, task in tasks.items():
    if type(task) is FrameTask:
        task.set_target(configuration.get_transform_frame_to_world(body))
```

A task can be added to the inverse kinematics once both its cost and target (if applicable) are defined.

### Differential inverse kinematics

Pinker solves differential inverse kinematics, meaning it outputs a velocity that steers the robot towards achieving all tasks at best. If we keep integrating that velocity, and task targets don't change over time, we will converge to a stationary configuration:

```python
dt = 6e-3  # [s]
for t in np.arange(0.0, 42.0, dt):
    velocity = solve_ik(configuration, tasks.values(), dt, solver="quadprog")
    configuration.integrate_inplace(velocity, dt)
    time.sleep(dt)
```

If task targets are continuously updated, there will be no stationary solution to converge to, but the model will keep on tracking each target at best. By default, `solve_ik` will take into account both joint limits and velocity limits read from the robot model.

## Examples

The `examples/` directory mirrors Pink's examples, ported to the `pinker.kinematics` backend with [Viser](https://viser.studio) visualization. Each one is named `<robot>_<task>.py`, after the robot description it loads and what it does with it:

```console
pixi run example examples/ur3_end_effector_tracking.py
```

Each example can also be run standalone with [uv](https://docs.astral.sh/uv/):

```console
uv run examples/ur3_end_effector_tracking.py
```

- **Single arms:** [Panda](https://github.com/pink-kinematics/pinker/tree/main/examples#panda-end-effector-tracking), [Panda maximizing manipulability](https://github.com/pink-kinematics/pinker/tree/main/examples#panda-manipulability), [UR5](https://github.com/pink-kinematics/pinker/tree/main/examples#ur5-end-effector-tracking), [UR5 with end-effector limits](https://github.com/pink-kinematics/pinker/tree/main/examples#ur5-position-barrier)
- **Humanoid:** [Draco 3](https://github.com/pink-kinematics/pinker/tree/main/examples#draco-3-reaching)
- **Mobile base:** [Stretch R1](https://github.com/pink-kinematics/pinker/tree/main/examples#stretch-world-target)
- **Quadruped:** [Go2 squatting with floating-base limits](https://github.com/pink-kinematics/pinker/tree/main/examples#go2-squat-barrier)
- **Floating base:** [Clamp free-flyer velocities](https://github.com/pink-kinematics/pinker/blob/main/examples/upkie_floating_base_velocity_limit.py)
- **Wheeled biped:** [Upkie rolling without slipping](https://github.com/pink-kinematics/pinker/tree/main/examples#upkie-rolling)

Check out the [examples](https://github.com/pink-kinematics/pinker/tree/main/examples) directory for more.

## Compatibility

Pinker is API-compatible with Pink 4.4.0, with the following exceptions:

- Default limits are now stored in `configuration.default_limits` rather than cached on the robot model.
- Functions and methods of the kinematics backend follow Python's PEP 8 naming conventions, e.g. `model.get_frame_id` instead of `getFrameId`.
- Integrating via `Configuration.integrate` returns a new `Configuration` rather than a configuration vector.
- Model getters raise rather than returning the sentinel index when a name is not found.
- Pink's `SelfCollisionBarrier` was not carried over, and configurations don't have a collision model. Use Pink if you need collision-avoidance tasks.
- Pinker works with a single visualizer, [Viser](https://viser.studio), used for both visualization and user inputs.

## Benchmark

Pinker and Pink were compared on a Raspberry Pi 4 Model B using the [pinker benchmark](https://github.com/pink-kinematics/pinker_benchmark), which runs both of them on the [pink motions](https://github.com/pink-kinematics/pink_motions/) library of robot trajectories.

Here are the results from running the benchmark on 2026-09-27 (aarch64, commit 751da6a89) comparing pinker 1.0.0 to pink 4.4.0 (pinocchio 4.1.0). QP solver is clarabel, 10 rollouts per scenario. The conclusions are that:

1. **Pinker produces the same IK problems as Pink:** ✅ (numerical variations less than 1e-09)
2. **Pinker has the same performance as Pink:** ✅ (timings variations less than 3%)

Here are the statistics scenario by scenario:

| scenario      | nv | max QP distance | IK check | Pink step (ms) | Pinker step (ms) | step var. (%) | perf check |
|:--------------|---:|----------------:|:---------|---------------:|-----------------:|--------------:|:-----------|
| edo           |  6 |           6e-15 | ✅       |    1.99 ± 0.01 |      1.98 ± 0.01 |          -0.6 | ✅         |
| fanuc         |  6 |           9e-14 | ✅       |    2.19 ± 0.01 |      2.18 ± 0.01 |          -0.4 | ✅         |
| gen2          |  6 |           5e-15 | ✅       |    2.14 ± 0.01 |      2.13 ± 0.01 |          -0.4 | ✅         |
| gen3          |  7 |           1e-14 | ✅       |    2.19 ± 0.01 |      2.17 ± 0.01 |          -0.6 | ✅         |
| iiwa14        |  7 |           3e-15 | ✅       |    2.24 ± 0.01 |      2.24 ± 0.01 |          -0.2 | ✅         |
| panda         |  9 |           1e-15 | ✅       |    2.39 ± 0.01 |      2.38 ± 0.01 |          -0.5 | ✅         |
| poppy_ergo_jr |  6 |           2e-15 | ✅       |    1.98 ± 0.01 |      1.97 ± 0.01 |          -0.5 | ✅         |
| ur10          |  6 |           4e-15 | ✅       |    2.17 ± 0.01 |      2.16 ± 0.01 |          -0.3 | ✅         |
| ur3           |  6 |           3e-15 | ✅       |    2.16 ± 0.01 |      2.15 ± 0.01 |          -0.5 | ✅         |
| ur5           |  6 |           3e-15 | ✅       |    2.16 ± 0.01 |      2.15 ± 0.01 |          -0.3 | ✅         |
| z1            |  6 |           2e-14 | ✅       |    2.16 ± 0.01 |      2.15 ± 0.01 |          -0.4 | ✅         |
| atlas_drc     | 36 |           3e-13 | ✅       |    3.89 ± 0.02 |      3.88 ± 0.02 |          -0.5 | ✅         |
| atlas_v4      | 36 |           3e-13 | ✅       |    3.89 ± 0.01 |      3.88 ± 0.01 |          -0.1 | ✅         |
| draco3        | 33 |           5e-14 | ✅       |    3.72 ± 0.01 |      3.72 ± 0.01 |          -0.1 | ✅         |
| ergocub       | 63 |           1e-14 | ✅       |    7.76 ± 0.03 |      7.64 ± 0.03 |          -1.6 | ✅         |
| h1            | 25 |           1e-14 | ✅       |    3.33 ± 0.01 |      3.35 ± 0.01 |          +0.6 | ✅         |
| icub          | 38 |           2e-13 | ✅       |    4.27 ± 0.01 |      4.20 ± 0.01 |          -1.7 | ✅         |
| jaxon         | 44 |           1e-13 | ✅       |    4.13 ± 0.01 |      4.11 ± 0.01 |          -0.6 | ✅         |
| jvrc          | 50 |           5e-13 | ✅       |    4.61 ± 0.02 |      4.58 ± 0.02 |          -0.5 | ✅         |
| r2            | 62 |           2e-14 | ✅       |    5.66 ± 0.02 |      5.61 ± 0.02 |          -0.9 | ✅         |
| romeo         | 67 |           5e-14 | ✅       |    5.08 ± 0.02 |      5.00 ± 0.02 |          -1.4 | ✅         |
| sigmaban      | 26 |           6e-14 | ✅       |    3.03 ± 0.01 |      3.05 ± 0.01 |          +0.6 | ✅         |
| talos         | 50 |           9e-12 | ✅       |    4.33 ± 0.02 |      4.30 ± 0.01 |          -0.6 | ✅         |
| valkyrie      | 65 |           4e-15 | ✅       |    5.44 ± 0.03 |      5.39 ± 0.03 |          -1.0 | ✅         |
| bolt          | 12 |           2e-15 | ✅       |    2.61 ± 0.01 |      2.65 ± 0.01 |          +1.8 | ✅         |
| cassie        | 22 |           3e-15 | ✅       |    3.34 ± 0.01 |      3.36 ± 0.01 |          +0.7 | ✅         |
| spryped       | 14 |           8e-15 | ✅       |    2.59 ± 0.01 |      2.63 ± 0.01 |          +1.7 | ✅         |

See the readme and data files in the benchmark repository for more details.

## Citation

If you use Pinker in your scientific works, please cite it *e.g.* as follows:

```bibtex
@software{pinker,
  title = {{Pinker: Python inverse kinematics for embedded robots}},
  author = {Caron, Stéphane and De Mont-Marin, Yann and Budhiraja, Rohan and Bang, Seung Hyeon and Domrachev, Ivan and Nedelchev, Simeon and Du, Peter and Escande, Adrien and Vaillant, Joris and Wingo, Bruce and Patapati, Santosh and San José Pro, Daniel and Marticorena Vidal, Nicolas Guillermo},
  license = {Apache-2.0},
  url = {https://github.com/pink-kinematics/pinker},
  version = {1.0.0},
  year = {2026}
}
```

Don't forget to add yourself to the BibTeX above and to `CITATION.cff` if you contribute to this repository.

## See also

Software:

- [Jink.jl](https://github.com/adubredu/Jink.jl): Julia package for differential multi-task inverse kinematics.
- [mink](https://github.com/kevinzakka/mink): differential inverse kinematics in Python, based on the MuJoCo physics engine.
- [Pink](https://github.com/pink-kinematics/pink/): precursor to Pinker based on Pinocchio.
- [Pink motions](https://github.com/pink-kinematics/pink_motions/): library of robot motions that can be used for benchmarking or continuous integration.
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio): C++ rigid body dynamics algorithms library and reference implementation for the C kinematics backend of Pinker.
- [PlaCo](https://github.com/rhoban/placo): C++ differential multi-task inverse kinematics based on Pinocchio.
- [pymanoid](https://github.com/stephane-caron/pymanoid): precursor to Pink and Pinker based on OpenRAVE.
- [TSID](https://github.com/stack-of-tasks/tsid): C++ inverse kinematics based on Pinocchio.

Technical notes:

- [Differential inverse kinematics](https://scaron.info/robotics/differential-inverse-kinematics.html)
- [Jacobian of a kinematic task and derivatives on manifolds](https://scaron.info/robotics/jacobian-of-a-kinematic-task-and-derivatives-on-manifolds.html)
- [Control Barrier Functions](https://web.archive.org/web/20241125170734/https://simeon-ned.com/blog/2024/cbf/)
