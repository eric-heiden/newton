# Fruit-bowl replay: file formats and the `replay_common` API

This file describes the data files of the workspace, their clocks, and the
helpers in `replay_common.py`. `replay_common.py` is fixed: verification uses its
own copy of it, so edits to the workspace copy have no effect there.

## Conventions

- **Units.** SI throughout: meters, seconds, radians. Gripper openings are
  0 (closed) to 1 (open). Quaternions are `(x, y, z, w)`.
- **World frame.** This is the station MJCF world (`station/yam_bimanual_empty.xml`):
  z points up and the table top is at z = 0.75.
- **Clocks.**
  - **State clock.** Times are seconds from the first top-camera frame. Arm
    states, commands and gripper logs (`<side>_t`, `<side>_cmd_t`, ...) use this
    clock. The canonical sample times are `left_t`.
  - **Camera clock.** Camera frame times (`t_top`, `t_left`, `t_right`) use the
    same origin.
  - **Time-base rule.** Top frame `i` shows the state at arm-state sample `i - 1`.
    Frame 0 shows sample 0. The same rule is assumed for the wrist cameras; this
    was checked visually only.
  - **Episodes with other frame rates.** An episode can set
    `top_state_index[i]` explicitly, for example for 10 fps episodes.
  - **Using the rule.** `frame_rows(episode, camera)` applies the rule. All
    metrics are computed on state samples.
  - **Sim time.** World `w` starts at its episode's `left_t[0]`. Physics step `k`
    applies the command schedule at control time `left_t[0] + k * dt`, so the
    state after step `k` is at `left_t[0] + (k + 1) * dt`.
- **Commands.** Logged commands are held (zero-order hold) from their timestamps
  and take effect after `command_delay`.
  - **Gripper.** The left finger slide target is
    `clip(grip_cmd, 0, 1) * 0.0475` m. The right finger is mirrored by the MJCF
    equality, and its target is set to the negative value.
  - **Initial state.** Arms start at the measured `q[0]` and `qd[0]` (`qd = 0` if
    the episode has none). Fingers start at the measured opening `grip[0]`.
- **Names.**
  - Fruits are the bodies labelled `pear`, `orange` and `dark_fruit`, one per
    world.
  - Station joints and bodies keep their MJCF names: `<side>_joint1..6`,
    `<side>_left_finger`, `<side>_right_finger`, the pad bodies
    `<side>_lf_down` / `<side>_rf_down`, the hand body `<side>_link_6`, and
    the wrist camera bodies `<side>_camera_frame`.
  - Matching uses the last component of each label.

## Workspace layout

```
fruit_replay.py       the replay script (edit this)
replay_common.py      fixed helpers (this file documents them)
station/              station MJCF (yam_bimanual_empty.xml) and meshes
episodes/main.npz     the main episode (about 30 Hz); episodes/sib_{1..4}.npz: four 10 fps episodes
scenes/main.json      the scene of each episode (same names)
camera.json           camera calibration
video_reference.npz   ground truth of the main episode from its video (format below)
gt/sib_{1..4}.npz     ground truth of the 10 fps episodes (same format)
frames/main/{top,left_wrist,right_wrist}/%03d.jpg, frames/sib_{1..4}/top/%03d.jpg
arm_logs/             64 recorded YAM arm logs (CSV, 32 episodes of various tasks, both arms), for arm calibration
```

`load_episode("main")` resolves to `episodes/main.npz` next to `replay_common.py`, and
`load_scene("main")` resolves to `scenes/main.json`. Explicit paths work everywhere.

Verification imports `fruit_replay.py` in a copy of this workspace without `frames/`, `arm_logs/`, `gt/`,
and `video_reference.npz`, with its own `replay_common.py`, station, episodes, and scenes. `build_model`
gets only scenes, and without `episode`, `uuid`, `source`, and `notes` (unseen episodes are named
`heldout`), so fitted values belong in the script or in a file of yours next to it.

## Episode `.npz`

`<side>` is `left` or `right`.

| key | shape | meaning |
|---|---|---|
| `t_top` | [F] | top-camera frame times [s] |
| `t_left`, `t_right` | [F] | wrist-camera frame times [s] (optional) |
| `top_state_index` | [F] int | optional: state row shown by each top frame (default `max(i - 1, 0)`) |
| `<side>_t` | [n] | arm-state times [s] (`left_t` is the canonical sample clock) |
| `<side>_q` | [n, 6] | measured joints [rad] |
| `<side>_qd` | [n, 7] | measured joint velocities [rad/s] (optional; column 6 is the gripper) |
| `<side>_tau` | [n, 7] | measured motor torques [N m] (optional) |
| `<side>_pose`, `<side>_cmd_pose` | [n, 4, 4] | logged end-effector poses (optional, informational) |
| `<side>_cmd_t`, `<side>_cmd` | [m], [m, 6] | joint commands [rad] at their timestamps [s] |
| `<side>_grip_t`, `<side>_grip` | [g], [g] | measured gripper opening [0..1] |
| `<side>_grip_cmd_t`, `<side>_grip_cmd` | [h], [h] | commanded gripper opening [0..1] |

**10 fps episodes** (`sib_1` .. `sib_4`) were built with
`episode_from_lerobot(timestamp, state, action, velocity=..., torque=..., time_offset=...)`.

- **Inputs.** LeRobot rows are `[left j1..j6, left grip, right j1..j6, right grip]`.
  Every row is a copy of the MCAP sample nearest to its 10 fps tick (checked on
  the LeRobot copy of the main episode: states, actions, velocities and torques
  match MCAP samples exactly).
- **Grid.** States, actions, velocities (`qd`) and torques (`tau`) are linearly
  interpolated onto a 30 Hz grid that starts at the first tick.
  `time_offset` (the state stream's start minus the top camera's, 0.097-0.115 s
  per episode from the LeRobot metadata) puts the grid on the main episode's
  clock, seconds from the first top frame.
- **Frames.** `t_top` holds the tick times on that clock. LeRobot top frame `k`
  is the MCAP top frame nearest to its tick, `3k + 3` (`3k + 2` after about 8 s),
  found by matching the main episode's two copies image by image. By the
  time-base rule it shows state sample `3k + 2`, so `top_state_index[k] = 3k + 2`:
  a 10 fps frame shows the state 2/30 s *after* its tick (`frame_lag = -2/30`).

## Scene `.json`

`scenes/main.json`, abridged:

```json
{
 "name": "main", "episode": "episodes/main.npz", "uuid": "7067ee1f-...",
 "table_z": 0.75,
 "bases": {"left": [0.2525, 0.31, 0.75], "right": [0.2546875, -0.300, 0.75]},
 "tray": {"apex_xy": [0.926, 0.1438], "yaw_deg": -161.0, "radius_m": 0.2362, "half_angle_deg": 31.5,
          "rim_height_m": 0.018, "floor_height_m": 0.005},
 "fruits": {
  "pear": {"arm": "left", "shape": "pear",
           "size_m": {"width": 0.051, "length": 0.081},
           "size_range_m": {"width": [0.044, 0.058], "length": [0.065, 0.092]},
           "centre_height_m": 0.0255, "mass_range_kg": [0.02, 0.15],
           "events": {"cmd_close": 40, "contact": 47, "liftoff": 53, "cmd_open": 79, "release": 80, "settled": 107},
           "start": {"pos": [0.65366, 0.32903, 0.7755], "quat_xyzw": [0, 0, -0.239541, 0.970886]},
           "start_image_xy": [0.6467, 0.3411]},
  "orange": {"arm": "left", "shape": "sphere", "size_m": {"diameter": 0.058}, "centre_height_m": 0.029, "...": "..."},
  "dark_fruit": {"arm": "right", "shape": "sphere", "size_m": {"diameter": 0.069}, "centre_height_m": 0.0345, "...": "..."}
 },
 "order": ["pear", "orange", "dark_fruit"],
 "time_base": "top frame i shows arm-state sample i-1"
}
```

- **`bases`.** These are the arm base positions. Apply them with
  `set_arm_bases(builder, scene["bases"])` on a one-station builder before
  `add_world`. The right base at y = −0.300 comes from the video fit; the MJCF
  has −0.31.
- **`events`.** These are top-frame indices of the real events, in the
  episode's own frame rate.
- **`centre_height_m`.** This is the resting centre height above the table. It
  is half the real held gripper gap. It is also the fruit radius `r` in the
  holding and placement tests.
- **`start`.** This is the FK-consistent rest pose, from
  `StationFK(station, scene["bases"]).fruit_starts(episode, scene)`.
  - **xy.** The midpoint of the grasping arm's two pad axes at the fruit's centre
    height, averaged over frames `contact .. liftoff - 2`.
  - **z.** `table_z + centre_height_m`. The builder may add clearance (the
    starter adds 0.5 mm).
  - **Orientation.** The pear's long axis is body x, perpendicular to the closing
    direction one frame before lift-off.
  - **Base placement.** Starts are computed with the scene's bases. The scene's
    right base moves the main episode's dark fruit +2.1 mm in x and +10 mm in y
    compared with the MJCF base.
- **`start_image_xy`.** This is the image-tracker start. It is informational and
  1.4 / 2.3 / 4.0 cm off the FK starts.
- **Other fields.** `grip_gap_m` (measured finger gap while held), `final_image_xyz`
  (final centre from the video, on the tray floor), `grasp_order`, `frame_rate`,
  `source`; the tray may carry its silhouette `iou`.

**10 fps scenes** are derived from the logs and the first and last top frames:
- the holds and their events from the gripper signals with the tracker's rules
  (they reproduce the main episode's tracker events exactly at 30 Hz); events are
  10 fps frame indices, `event_rows` the 30 Hz state rows they came from;
- the fruit of each hold from the first-frame colour detections nearest to the
  pad midpoint, confirmed by hand in the wrist views;
- `centre_height_m` is the nominal value of the main episode for every episode
  (the same fruits), not half that episode's held gap;
- the tray sector from the first-frame silhouette, with the main episode's radius
  and half-angle (on the main episode's LeRobot copy: 3 mm and 1.1 deg from the
  tracker's fit);
- `notes` with what a check by hand of the wrist and top views found.
- **Ensembles.** `jitter_scene(scene, rng)` perturbs the fruit starts the way the
  verification ensembles do: xy moves by N(0, 2 mm) per axis, clipped at 4 mm, and
  the pear yaw changes by U(±5°).

## Ground truth `.npz` (`load_gt`)

Per fruit `<f>`:

| key | shape | meaning |
|---|---|---|
| `<f>_pos` | [F, 3] | fruit centre per top frame [m], NaN where unknown; FK-attached while carried |
| `<f>_src` | [F] int8 | 1 image fix, 2 gripper FK (attached), 3 interpolated, 4 held rest (optional) |
| `<f>_phase` | [F] int8 | 0 rest, 1 grasped, 2 carried, 3 released, 4 resting, 5 pushed (optional) |
| `<f>_uv`, `<f>_fix`, `<f>_radpx` | [F, 2], [F], [F] | image centroid [px], unoccluded fix flag, image radius [px] (optional; image metric) |
| `<f>_rest_xyz`, `<f>_final_xyz` | [3] | real start and final centre [m] |
| `<side>_pad`, `<side>_gap`, `<side>_grip` | [F, 3], [F], [F] | FK pad centre, pad gap, gripper state (optional) |

- **Main episode** (`video_reference.npz`, from a video tracker). Every carry
  frame of `pos` is FK-attached, using an image-fitted contact offset.
- **10 fps episodes** (`gt/sib_k.npz`, and the same for the unseen verification
  episodes). `pos` is
  `StationFK(...).attached_tracks(episode, scene)`: the FK start before contact
  (`src` 4), the hand-attached centre from contact to release (`src` 2), NaN after
  release, and the last frame's image position (`src` 1). Its offset comes from
  the FK start, not the image. On the main episode it differs from the tracker's
  carry track by a median of 1.2 / 2.1 / 4.0 cm. `rest_xyz` is the first frame's
  image position (`start_fk_xyz` the FK start), `final_xyz` the last frame's;
  `uv`/`fix` hold the detections of those two frames, and `t`, `frame`,
  `top_state_index` the frame clock.
- **Without ground truth.** When there is no carry z, `lifted_fraction` falls
  back to "at least 2 cm above the start".

## `camera.json`

```json
{
 "top":   {"width": 640, "height": 480, "K": [9 values], "D": [k1, k2, p1, p2, k3],
           "distortion_model": "inverse_brown_conrady", "position": [x, y, z], "rotation_xyzw": [x, y, z, w]},
 "left":  {"width": 640, "height": 480, "K": [...], "D": [...], "distortion_model": "inverse_brown_conrady",
           "body": "left_camera_frame", "body_rotation_xyzw": [1, 0, 0, 0]},
 "right": {"...": "same as left with right_camera_frame"},
 "time_base": "top frame i shows arm-state sample i-1"
}
```

- **Top camera.** The pose is fitted to the main episode's video. The camera
  looks along its −Z axis with +Y up.
- **Wrist cameras.** Each follows its MJCF body. The camera pose is
  `body_q[body] * rotation(1, 0, 0, 0)`. Applying the same rotation to
  `top_camera_frame` gives the CAD top-camera pose.
- **Intrinsics.** These are the RealSense intrinsics of the recording.

## Recording (one world, `Replay.recordings()`)

Rows are the episode's state samples `left_t` up to the current step. Values are
interpolated from the device history (recorded every `round(record_dt / dt)` steps,
default 1/120 s): linearly for positions and joints, and with normalized lerp for
rotations.

| key | shape | meaning |
|---|---|---|
| `t` | [n] | state times [s] |
| `top_row` | [F] | row shown by each top frame (`frame_rows`) |
| `<f>_pos`, `<f>_quat` | [n, 3], [n, 4] | fruit pose |
| `<side>_pad` | [n, 3] | midpoint of the two pads' grasp points (pad body point `(0, -0.0024, 0.071)`, 25 mm toward the tip from the pad box centre) |
| `<side>_hand_pos`, `<side>_hand_quat` | [n, 3], [n, 4] | `<side>_link_6` pose (frame for slip and release) |
| `<side>_finger` | [n] | left finger slide [m] |
| `<side>_q` | [n, 6] | simulated arm joints [rad] (`state.joint_q`; the solver must maintain it) |
| `<side>_q_real`, `<side>_grip_real` | [n, 6], [n] | measured joints and opening at the same rows |
| `body_q`, `body_index` | [n, B, 7], [B] | all body poses of the world and their global indices (for `recording_state` + rendering) |
| `world`, `complete` | scalars | world index; whether all state samples were simulated |

## `replay_common` API

**Data.**
- `load_episode(path_or_name)` loads an episode.
- `episode_from_lerobot(timestamp, state, action, velocity=None, torque=None, time_offset=0.098, rate=30, frame_lag=-2/30)`
  builds an episode from a LeRobot copy.
- `frame_rows(episode, camera="top")` gives the state row shown by each frame.
- `measured(episode, side)` returns the measured `(q, grip)` on `left_t`.
- `load_scene(path_or_name)` loads a scene.
- `load_gt(path)` and `gt_from_tracker(meta, trajectories)` load ground truth.
- `load_camera(path=None)` loads camera.json.
- `jitter_scene(scene, rng, xy_sigma=0.002, xy_clip=0.004, yaw_deg=5)` perturbs fruit starts.

**Geometry.**
- `set_arm_bases(builder, bases)` places the arm bases.
- `sector_distance(xy, tray)` returns the distance to the tray footprint, 0 inside.
- `sector_polygon(tray)` returns the tray outline.
- `quat_to_matrix(q)` converts quaternions to rotation matrices.
- `StationFK(station_xml, bases)` evaluates forward kinematics on the CPU:
  - `pose(arm_q, grip)` and `pose_row(episode, row)` return body poses.
  - `pads(body_q, side)` and `grasp_point(body_q, side)` return pad geometry.
  - `fruit_starts(episode, scene)` and `attached_tracks(episode, scene)` produce
    fruit starts and FK-attached carry tracks.

**Commands.** `command_schedule(episode, times, delay)` returns
`{side: {"q": [T, 6], "finger": [T]}}`.

**Replay.** `Replay(model, solver, pipeline, episodes, dt=..., command_delay=0.0, contacts=None, record_dt=1/120, use_graph=True)`
takes one episode per world, or a single episode for all worlds. `command_delay`
is a scalar or one value per world.
- **One step.** Each step does the following:
  1. A kernel writes all worlds' arm and finger targets from the dense schedule
     at row `min(step, T - 1)`. After an episode ends, its last command is held.
  2. `pipeline.collide(state_0, contacts)`.
  3. `solver.step(state_0, state_1, control, contacts, dt)`.
  4. `state_0.assign(state_1)`.
  5. A record kernel writes the history row.
  6. A separate one-thread kernel advances the cursor `step_index`. It never
     reads and increments the cursor in the same launch.
- **CUDA graph.** The step is captured as a CUDA graph after one warm-up step
  and a reset. Without a graph, or on CPU, steps run in a host loop. Both paths
  agree to 1.5e-8 over 0.56 s.
- **Methods.**
  - `reset()` rewinds the replay. It writes the initial state and first targets,
    sets the cursor to 0, calls `solver.reset(state, flags=StateFlags.NONE)`, and
    resets pipeline contact matching.
  - `step(n)` advances `n` steps.
  - `run(seconds=None)` steps to `seconds`, or to the end, and returns the wall
    time.
  - `capture()` re-records the graph. Call it after replacing `replay.solver`,
    `collision_pipeline` or `contacts`.
- **Properties.** `steps_done`, `sim_time`, `done` and `total_steps`
  (`total_steps` is rounded up to whole recording periods).
- **Recordings.** `recordings(worlds=None)` returns a list and `recording(world)`
  returns one world.
- **Rendering a recorded row.** `recording_state(model, recording, row, state=None)`
  builds a state for rendering that row.

**Scoring.**
- `score(recording, scene, gt=None, camera=None)` returns
  `{"complete", "arm_rmse_rad": {side}, "fruits": {name: metrics}, "all_held", "all_placed"}`.
- `check(metrics, thresholds=None)` returns `{"pass", "failed": [...]}` using
  `DEFAULT_THRESHOLDS`, the per-rollout counterparts of the verification gates.
- `summarize([metrics, ...])` returns, for an ensemble of one episode, per-fruit
  held and placed counts and medians.

**Cameras.**
- `StationRenderer(model, camera=None, supersample=1, look=None)` provides:
  - `render_top(state, world=0, masks=False)`
  - `render_wrist(state, side, world=0, masks=False)`
  - `render(state, camera_name, pose, world, masks)`
  - `top_pose()` and `wrist_pose(state, side, world)`
- `render_top(model, state, world=0, camera=None, masks=False)` and
  `render_wrist(model, state, side, ...)` cache one renderer per model.
- Images are [480, 640, 3] uint8 sRGB. With `masks=True` the call also returns
  `{fruit: bool [480, 640]}` visible pixels.
- `project_points(intrinsics, points, pose=None)` projects world points to pixels
  through the distortion model.
- `camera_rays(intrinsics, supersample)` returns the camera rays.

**Contacts.** `contact_summary(model, state, contacts, a, b=None, solver=None, world=None)`.
- **Selectors.** These are label substrings of a shape or its body (for example
  `"left_lf_down"`, `"pear"`, `"table_plane"`, `"tray"`), shape indices, or
  lists of these.
- **Returned values.** `count`, `touching`, `normal_force` [N],
  `tangential_force` [N], `penetration` [m], `slip_max` [m/s] and `by_body`.
- **Source of contacts and forces.**
  - With `SolverMuJoCo`, pass `solver=`. The contact set and forces are then the
    solver's own.
  - Otherwise the `contacts` buffer is read, and forces are `nan` unless the
    solver implements `update_contacts`.

### Example integration

Tools that snapshot an example's own Warp arrays and scalar attributes
(checkpoints) rewind the replay too when the example aliases the replay's arrays
and state. The starter's `Example` does this:

```python
self.replay = replay_common.Replay(
    self.model, self.solver, self.collision_pipeline, episodes, dt=PARAMS["dt"], command_delay=PARAMS["command_delay"]
)
self.state_0, self.state_1 = self.replay.state_0, self.replay.state_1
self.control, self.contacts = self.replay.control, self.replay.contacts
for name, array in self.replay.checkpoint_arrays().items():  # replay_step, replay_history_*
    setattr(self, name, array)
self.graph = self.replay.graph


def capture(self):
    self.replay.solver, self.replay.collision_pipeline = self.solver, self.collision_pipeline
    self.replay.capture()
    self.graph = self.replay.graph


def step(self):  # one example frame
    self.replay.step(self.substeps)
    self.sim_time += self.frame_dt
```

Call `example.capture()` after replacing `solver`, `collision_pipeline` or
`contacts`. `replay.reset()` returns the cursor to 0.

## Metrics (`score`)

All metrics use state-sample rows. Event frames are converted with
`top_row`. The carry window runs from `row(liftoff) + 3` to `row(release) - 2`,
inclusive.

| metric | definition |
|---|---|
| `moved_before_grasp_m` | max xy displacement from the start, up to `row(cmd_close)` |
| `max_rise_m` | highest centre height above the start over the episode |
| `held_fraction` (`held` ≥ 0.9) | carry rows with \|fruit − pad midpoint of its arm\| ≤ r + 15 mm |
| `lifted_fraction` | carry rows with sim rise ≥ 0.5 × real rise (GT `pos`) |
| `liftoff_err_s` (`_rows`) | first row after `cmd_close` with the centre 1 cm above its start, minus `row(liftoff)` |
| `release_err_s` | first carry row with the fruit 1 cm from its in-hand (link_6 frame) position at the carry start, minus `row(release)` |
| `carry_track_err_m` | median xy distance to GT `pos` over carry rows (`carry_track_err_3d_m` in 3D, report only) |
| `grip_gap_err_mm` | median of 2 × (sim slide − measured opening × 0.0475), from `row(liftoff)` to `row(cmd_open)` |
| `slip_m` | max drift in the hand frame over the carry |
| `placed` | complete; sector distance ≤ 10 mm; \|z − (table + floor + r)\| ≤ 15 mm; mean speed over the last 0.3 s < 2 cm/s |
| `arm_rmse_rad` | per arm, over all rows |
| `final_xy_err_m` | to GT `<f>_final_xyz` |
| `image_err_px_post` (report) | median projected-centroid error on fix frames after release (needs `camera`) |

- **Per-rollout check.** `check()` compares one scored rollout with
  `DEFAULT_THRESHOLDS`, the single-rollout counterparts of the verification
  gates. Verification itself gates ensembles of jittered copies (counts of copies
  and medians over copies; see the task description).
- **10 fps episodes.** Their ground-truth carry tracks have a third of the main
  episode's samples and their event frames are 100 ms apart, so `lifted_fraction`
  and the timing metrics are coarse there. On 10 fps episodes verification gates
  only whether fruits are held and placed, and arm tracking.
