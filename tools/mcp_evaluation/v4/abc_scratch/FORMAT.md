# Workspace files: formats and frames

This file describes the data files of the workspace, their clocks, and their
coordinate frames. It describes the data only.

## Conventions

- **Units.** SI throughout: meters, seconds, radians, newton meters. Gripper
  openings run from 0 (closed) to 1 (open). Quaternions are `(x, y, z, w)`.
- **World frame.** This is the world of the station MJCF
  (`station/yam_bimanual_empty.xml`). z points up, and the table top is at
  z = 0.75 m.
- **State clock.** All times are seconds on one clock. Arm states, commands, and
  gripper logs (`<side>_t`, `<side>_cmd_t`, ...) use it. The canonical sample
  times are `left_t`, and a "state row" is an index into `left_t`.
- **Photos and state rows.** Top-camera frame `i` of the recording shows state
  row `max(i - 1, 0)`. The wrist cameras are assumed to follow the same rule
  (checked visually only). `photos/index.json` gives the row each photo shows.
- **Names.** Station joints and bodies keep their MJCF names:
  - arm joints `<side>_joint1` .. `<side>_joint6`;
  - finger slides `<side>_left_finger` and `<side>_right_finger`, coupled by an
    MJCF equality;
  - finger pad bodies `<side>_lf_down` and `<side>_rf_down`;
  - hand body `<side>_link_6`;
  - wrist camera bodies `<side>_camera_frame`.

  `<side>` is `left` or `right`. In a Newton model, the last component of a
  label (after the last `/`) is the MJCF name.

## Workspace layout

```
scene_replay.py   the replay script (edit this)
FORMAT.md         this file
episode.npz       the recorded episode (about 30 Hz)
photos/           JPEG frames of the top and wrist cameras, with photos/index.json
camera.json       camera calibration
station/          the station MJCF (yam_bimanual_empty.xml) and its meshes
arm_logs/         64 recorded YAM arm logs (CSV)
```

## `episode.npz`

Every array has one row per sample.

| key | shape | meaning |
|---|---|---|
| `<side>_t` | [n] | arm-state times [s] (`left_t` is the canonical sample clock) |
| `<side>_q` | [n, 6] | measured joint positions, joints 1-6 [rad] |
| `<side>_qd` | [n, 7] | measured joint velocities [rad/s]; column 6 is the gripper motor |
| `<side>_tau` | [n, 7] | measured motor torques [N m]; column 6 is the gripper motor's effort |
| `<side>_cmd_t`, `<side>_cmd` | [m], [m, 6] | logged joint position commands [rad] at their timestamps [s] |
| `<side>_grip_t`, `<side>_grip` | [g], [g] | measured gripper opening [0..1] |
| `<side>_grip_cmd_t`, `<side>_grip_cmd` | [h], [h] | logged gripper commands [0..1] |

The starter's docstring states how `scene_replay.py` turns the logged commands
into joint position targets and where the arms and fingers start.

## `photos/`

`photos/index.json` lists the photos in time order:

```json
[{"file": "01_top.jpg", "camera": "top", "time_s": 0.0, "state_row": 0, "state_time_s": 0.0968,
  "shows": "start"}, ...]
```

| field | meaning |
|---|---|
| `file` | image file in `photos/` (640x480 JPEG) |
| `camera` | `top`, `left` (left wrist), or `right` (right wrist) |
| `time_s` | frame time of the camera stream [s], on the state clock |
| `state_row` | state row the photo shows (the rule above) |
| `state_time_s` | `left_t[state_row]` [s] |
| `shows` | what the photo shows |

## `camera.json`

```json
{
 "top":   {"width": 640, "height": 480, "K": [9 values], "D": [k1, k2, p1, p2, k3],
           "distortion_model": "inverse_brown_conrady", "position": [x, y, z], "rotation_xyzw": [x, y, z, w]},
 "left":  {"width": 640, "height": 480, "K": [...], "D": [...], "distortion_model": "inverse_brown_conrady",
           "body": "left_camera_frame", "body_rotation_xyzw": [1, 0, 0, 0]},
 "right": {"...": "the same as left, with right_camera_frame"},
 "convention": "...", "time_base": "..."
}
```

- **Intrinsics.** `K` is the row-major 3x3 camera matrix [px] of the recording.
  `D` holds the coefficients of RealSense's inverse Brown-Conrady model
  (librealsense `RS2_DISTORTION_INVERSE_BROWN_CONRADY`). Its Brown-Conrady
  polynomial maps the normalized coordinates of a recorded (distorted) pixel to
  undistorted ones, that is, from a pixel to its ray.
- **Top camera.** `position` [m] and `rotation_xyzw` give its pose in the world
  frame. The camera looks along its -Z axis, with +Y up in the image.
- **Wrist cameras.** Each one moves with its MJCF body (`body`). Its pose is the
  body's pose followed by the rotation `body_rotation_xyzw` (no offset). The
  camera then looks along its -Z axis with +Y up, like the top camera.

## `station/`

This is the ABC simulator's MJCF of the bimanual YAM station. It holds the two
6-DoF arms with parallel grippers (position actuators, finger equality), the
table (a collision plane at z = 0.75 m), the enclosure walls, the camera mounts,
and the meshes. It has no objects.

## `arm_logs/`

These are 64 CSV files, one arm each, from 32 recorded episodes of various tasks
at YAM stations. Every row has these columns:

```
time, q1..q6, qd1..qd6, tau1..tau6, cmd1..cmd6, grip, grip_cmd
```

- `time` [s] starts at 0 in each file.
- `q`, `qd`, and `tau` are measured joint positions [rad], velocities [rad/s],
  and motor torques [N m].
- `cmd` holds the commanded joint positions [rad].
- `grip` and `grip_cmd` are the measured and commanded gripper openings [0..1].

Both arms are the same model as the station's arms.
