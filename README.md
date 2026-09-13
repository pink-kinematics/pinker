# Pinker

[![Build](https://img.shields.io/github/actions/workflow/status/pink-kinematics/pinker/ci.yml?branch=main)](https://github.com/pink-kinematics/pinker/actions)
[![Documentation](https://img.shields.io/github/actions/workflow/status/pink-kinematics/pinker/docs.yml?branch=main&label=docs)](https://pink-kinematics.github.io/pinker/)

**P**ython **in**verse **k**inematics for **e**mbedded **r**obots.

Pinker is a leaner version of [Pink](https://github.com/pink-kinematics/pink/) for single-board computers. Two dependencies, one C file, and it takes 5 seconds to build from source on a Raspberry Pi 4. But it doesn't implement collision avoidance.

## Installation

You can install the library from PyPI:

```console
pip install pinker
```

You can also clone the repository and run it locally:

```bash
git clone https://github.com/pink-kinematics/pinker.git && cd pinker
uv run examples/humanoid_g1_com.py
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
from pinker import Configuration, load_robot, solve_ik

robot = load_robot("ur3_official_description")  # or: load_robot("robot.urdf")
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

## Compatibility

Pinker is API-compatible with **Pink 4.4.0**, with the following exceptions:

- Default limits live on the configuration, as `configuration.default_limits`, rather than being cached on the robot model. Add your own, for instance a `FloatingBaseVelocityLimit`, by appending to that list.
- `Configuration.integrate` returns a new `Configuration` rather than a configuration vector. Its vector is `configuration.integrate(v, dt).q`.
- Functions and methods of the kinematics backend follow Python naming, so `model.getFrameId` is `model.get_frame_id` and `model.lowerPositionLimit` is `model.lower_position_limit`. Model getters raise rather than returning the sentinel index Pinocchio returns when a name is not found.

Pinker is a standalone replacement for Pink where kinematics are carried out by `pinker.kinematics`, a backend written as a single C extension with a thin Python layer. Tasks, limits, barriers, the `Configuration` class and `solve_ik`, is the same as in Pink, and differential IK problems are still solved through [qpsolvers](https://github.com/qpsolvers/qpsolvers).

## Examples

The `examples/` directory mirrors Pink's examples, ported to the `pinker.kinematics` backend with [Viser](https://viser.studio) visualization:

```console
pixi run -e examples python examples/arm_ur3.py
```

Each example can also be run standalone with [uv](https://docs.astral.sh/uv/):

```console
uv run examples/arm_ur3.py
```

- **Single arms:** [Panda](https://github.com/pink-kinematics/pinker/tree/main/examples#arm-panda), [UR5](https://github.com/pink-kinematics/pinker/tree/main/examples#arm-ur5), [UR5 with end-effector limits](https://github.com/pink-kinematics/pinker/tree/main/examples#barrier-arm-ur5)
- **Humanoid:** [Draco 3](https://github.com/pink-kinematics/pinker/tree/main/examples#humanoid-draco-3)
- **Mobile base:** [Stretch R1](https://github.com/pink-kinematics/pinker/tree/main/examples#mobile-stretch)
- **Quadruped:** [Go2 squatting with floating-base limits](https://github.com/pink-kinematics/pinker/tree/main/examples#barrier-quadruped-go2)
- **Floating base:** [Clamp free-flyer velocities](https://github.com/pink-kinematics/pinker/blob/main/examples/floating_base_velocity_limit.py)
- **Wheeled biped:** [Upkie rolling without slipping](https://github.com/pink-kinematics/pinker/tree/main/examples#wheeled-biped-upkie)

Check out the [examples](https://github.com/pink-kinematics/pinker/tree/main/examples) directory for more.

## Limitations

- No collision support: Pink's `SelfCollisionBarrier` and the collision
  arguments of `Configuration` are not available. Use Pink if you need
  collision-avoidance tasks.
- One visualizer: Pinker works with [Viser](https://viser.studio), which
  handles both visualization and user inputs. If you would rather use (the
  older) MeshCat, head over to Pink, which is compatible with it.
- The `pinker.kinematics` backend is not type-checked yet: mypy is disabled on
  it in `pyproject.toml`. Enabling it is a matter of shipping a
  `_kinematics_c.pyi` stub for the C extension, annotating the arrays cached in
  `Model._packed` and the optional values the URDF parser reads, then removing
  the override.

## Benchmark

Pinker and Pink can be compared using [pinker_benchmark](https://github.com/pink-kinematics/pinker_benchmark), a standalone pixi project that runs a collection of arm and humanoid examples. On a Raspberry Pi 4 Model B:

```
TODO: benchmark results
```

See the readme and data files in the benchmark repository for more details.

## Citation

If you use Pinker in your scientific works, please cite it *e.g.* as follows:

```bibtex
@software{pinker,
  title = {{Pinker: Python inverse kinematics for embedded robots}},
  author = {Caron, Stéphane and De Mont-Marin, Yann and Budhiraja, Rohan and Bang, Seung Hyeon and Domrachev, Ivan and Nedelchev, Simeon and Du, Peter and Escande, Adrien and Vaillant, Joris and Wingo, Bruce and Patapati, Santosh and San José Pro, Daniel and Marticorena Vidal, Nicolas Guillermo},
  license = {Apache-2.0},
  url = {https://github.com/pink-kinematics/pinker},
  version = {0.1.0},
  year = {2026}
}
```

Don't forget to add yourself to the BibTeX above and to `CITATION.cff` if you contribute to this repository.

## See also

Software:

- [Jink.jl](https://github.com/adubredu/Jink.jl): Julia package for differential multi-task inverse kinematics.
- [mink](https://github.com/kevinzakka/mink): differential inverse kinematics in Python, based on the MuJoCo physics engine.
- [Pink](https://github.com/pink-kinematics/pink/): precursor to Pinker based on Pinocchio.
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio): C++ rigid body dynamics algorithms library and reference implementation for the C kinematics backend of Pinker.
- [PlaCo](https://github.com/rhoban/placo): C++ differential multi-task inverse kinematics based on Pinocchio.
- [pymanoid](https://github.com/stephane-caron/pymanoid): precursor to Pink and Pinker based on OpenRAVE.
- [TSID](https://github.com/stack-of-tasks/tsid): C++ inverse kinematics based on Pinocchio.

Technical notes:

- [Inverse kinematics](https://scaron.info/robotics/inverse-kinematics.html): a general introduction to differential inverse kinematics.
- [Jacobian of a kinematic task and derivatives on manifolds](https://scaron.info/robotics/jacobian-of-a-kinematic-task-and-derivatives-on-manifolds.html).
- [Control Barrier Functions](https://web.archive.org/web/20241125170734/https://simeon-ned.com/blog/2024/cbf/).
