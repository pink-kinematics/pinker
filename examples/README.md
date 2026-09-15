# Examples

Examples are named `<robot>_<task>.py`, after the robot description they load
and what they do with it. The following ones include *tasks* and *limits*:

- [UR5: end-effector tracking](#ur5-end-effector-tracking)
- [Panda: end-effector tracking](#panda-end-effector-tracking)
- [Draco 3: reaching](#draco-3-reaching)
- [UR10: inverse kinematics](#ur10-inverse-kinematics)
- [Stretch: mobile manipulation](#stretch-mobile-manipulation)
- [Upkie: rolling](#upkie-rolling)

The following ones include *control barrier functions*, see [this
note](https://web.archive.org/web/20241125170734/https://simeon-ned.com/blog/2024/cbf/) for an introduction:

- [UR5: position barrier](#ur5-position-barrier)
- [Go2: squat barrier](#go2-squat-barrier)

## UR5: end-effector tracking

In `ur5_end_effector_tracking.py`, a UR5 arm tracks a moving target:

https://github.com/stephane-caron/pink/assets/1189580/d0d6aae9-326b-45fe-8cd3-013f29f7343a

| Task         | Cost      |
|--------------|-----------|
| End-effector | 1         |
| Posture      | $10^{-3}$ |

## Panda: end-effector tracking

In `panda_end_effector_tracking.py`, a Panda arm tracks an interactive target in Viser:

https://github.com/user-attachments/assets/1c4ac222-8e3f-469d-95c3-550f1c0979fa

## Draco 3: reaching

In `draco3_reaching.py`, a Draco 3 humanoid moves its right hand laterally while standing. This model includes a closed kinematic chain, implemented in the example with a ``JointCouplingTask``:

https://github.com/stephane-caron/pink/assets/1189580/db6acda8-82a4-4f4d-9acf-1fc3d831e222

| Task | Cost |
|------|------|
| Left foot | (1, 1) |
| Left knee coupling | 100 |
| Posture | $10^{-1}$ |
| Right foot | (1, 1) |
| Right knee coupling | 100 |
| Right wrist | (4, 4) |
| Torso | (1, 0) |

## UR10: inverse kinematics

In `ur10_inverse_kinematics.py`, a UR10 arm solves inverse kinematics (by iterating differential IK) to find a configuration that achieves a given end-effector pose:

```console
$ uv run ur10_inverse_kinematics.py
Starting from error_norm = 2.4
Desired precision is error_norm < 1e-08
Terminated after 141 steps with error_norm = 1.1e-09
```

| Task         | Cost |
|--------------|------|
| End-effector | 1    |

## Stretch: mobile manipulation

In `stretch_mobile_manipulation.py`, a Stretch RE1 moves with a fixed fingertip target around the origin:

https://github.com/stephane-caron/pink/assets/1189580/711c4b92-6234-41bd-945b-e6c043f6b2e6

| Task | Position cost | Orientation cost |
|------|---------------|------------------|
| Mobile base | $0.1$ | 1 |
| Fingertip | 1 | $10^{-4}$ |

## Upkie: rolling

In `upkie_rolling.py`, an Upkie wheeled biped rolls without slipping:

https://github.com/user-attachments/assets/18ae0b68-21a2-44ec-af48-1d8ab4a7e658

| Task | Position cost | Orientation cost |
|------|---------------|------------------|
| Floating base | $1$ | $1$ |
| Left wheel rolling | $10$ | - |
| Right wheel rolling | $10$ | - |
| Left wheel position | $1$ | $0$ |
| Right wheel position | $1$ | $0$ |

## UR5: position barrier

In `ur5_position_barrier.py`, a UR5 arm tracks a moving target while stopping in front of a virtual wall:

https://github.com/domrachev03/pink/assets/28687492/f30ba7a1-98a3-44cb-ab52-23f99e42714c

| Task | Cost |
|------|------|
| End-effector | (10, 1) |
| Posture | $10^{-3}$ |

| Barrier | Gain |
|---------|------|
| End-effector position | $10^{2}$ |

## Go2: squat barrier

In `go2_squat_barrier.py`, a Go2 quadruped squats, with its base position constrained along the y- and
z-axes:

https://github.com/domrachev03/pink/assets/28687492/78281f44-3676-4d4d-9619-768b951a15a2

| Task | Cost |
|------|------|
| Base | (50, 1) |
| Feet | (200, 0) |
| Posture | $10^{-5}$ |

| Barrier | Gain |
|---------|------|
| Base position | $10^{2}$ |
