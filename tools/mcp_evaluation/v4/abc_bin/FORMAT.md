# Screwdriver-in-bin replay: file formats and the `replay_common` API

This file describes the data files of the workspace, their clocks, and the
helpers in `replay_common.py`. `replay_common.py` is fixed: verification uses its
own copy of it, so edits to the workspace copy have no effect there.

## Conventions

- **Units.** SI throughout: meters, seconds, radians, kilograms, newtons. Gripper
  openings are 0 (closed) to 1 (open). Quaternions are `(x, y, z, w)`.
- **World frame.** This is the station MJCF world (`station/yam_bimanual_empty.xml`):
  z points up and the table top is at z = 0.75.
- **Clocks.**
  - **State clock.** Times are seconds from the first top-camera frame. Arm
    states, commands and gripper logs (`<side>_t`, `<side>_cmd_t`, ...) use this
    clock. The canonical sample times are `left_t`.
  - **Camera clock.** Camera frame times (`t_top`, `t_left`, `t_right`) use the
    same origin.
  - **Time-base rule.** Top frame `i` shows the state at arm-state sample `i - 1`.
    Frame 0 shows sample 0. The wrist cameras are assumed to follow the same rule.
  - **Episodes with other frame rates.** An episode can set
    `<camera>_state_index[i]` explicitly (`top_state_index`, `right_state_index`,
    ...), as the 10 fps episodes do.
  - **Using the rule.** `frame_rows(episode, camera)` applies the rule. All
    metrics are computed on state samples.
  - **Sim time.** World `w` starts at its episode's `left_t[0]`. Physics step `k`
    applies the command schedule at control time `left_t[0] + k * dt`, so the
    state after step `k` is at `left_t[0] + (k + 1) * dt`.
- **Commands.** Logged commands are held (zero-order hold) from their timestamps
  and take effect after `command_delay`.
  - **Gripper.** The left finger slide target is
    `clip(grip_cmd, 0, 1) * 0.0475` m. The right finger is mirrored by the MJCF
    equality, and its target is set to the negative value. The measured opening
    `grip` maps to the slide the same way, so the measured jaw gap is
    `2 * clip(grip, 0, 1) * 0.0475` m.
  - **Initial state.** Arms start at the measured `q[0]` and `qd[0]` (`qd = 0` if
    the episode has none). Fingers start at the measured opening `grip[0]`.
- **Gripper effort.** Column 6 of `<side>_tau` is the gripper motor's measured
  effort τ [N·m] (columns 0-5 are the arm joints). The YAM gripper is i2rt's
  `linear_4310` gripper. For it, the clamping force F [N] that the jaws exert
  follows from the effort as F = (τ − 0.3) · 6.57 / 0.096, with τ in N·m. This
  holds while the jaws are closed on an object.
- **Names.**
  - The screwdriver is the body labelled `screwdriver`, one per world. A movable
    bin is the body labelled `bin`; a static bin has no body, and its shapes are
    labelled `bin...`.
  - Station joints and bodies keep their MJCF names: `<side>_joint1..6`,
    `<side>_left_finger`, `<side>_right_finger`, the pad bodies
    `<side>_lf_down` / `<side>_rf_down`, the hand body `<side>_link_6`, and
    the wrist camera bodies `<side>_camera_frame`.
  - Matching uses the last component of each label.
- **Screwdriver body frame.** The metrics read the pose of body `screwdriver` in
  the scene's object frame: the origin is the grasp point on the axis, and body +x
  runs along the axis toward the tip. Points such as `tip_local` and `com_local`
  are given in this frame.

## Workspace layout

```
screwdriver_replay.py   the replay script (edit this)
replay_common.py        fixed helpers (this file documents them)
FORMAT.md               this file
station/                station MJCF (yam_bimanual_empty.xml) and meshes
episodes/main.npz       the main episode (about 30 Hz); episodes/sib_{1..4}.npz: four 10 fps episodes
scenes/main.json        the scene of each episode (same names)
gt/main.npz             ground truth of each episode (same names)
camera.json             camera calibration
frames/main/{top,right_wrist}/%03d.jpg, frames/sib_{1..4}/{top,right_wrist}/%03d.jpg   recorded video, 848x480
arm_logs/               recorded YAM arm logs (CSV, various tasks, both arms), for arm calibration
```

Frame folder `right_wrist` belongs to camera `right` (`t_right`, `camera.json["right"]`).

`load_episode("main")` resolves to `episodes/main.npz` next to `replay_common.py`,
`load_scene("main")` to `scenes/main.json`, and `load_gt("main")` to `gt/main.npz`.
Explicit paths work everywhere.

Verification imports `screwdriver_replay.py` in a copy of this workspace without `frames/`, `arm_logs/`,
and `gt/`, with its own `replay_common.py`, station, episodes, scenes, and camera calibration. `build_model`
gets only scenes. These scenes lack `episode`, `uuid`, `source`, `notes`, `verdict`, and every key that starts
with `start_image_`, `wrist_`, or `final_`, and unseen episodes are named `heldout`. Fitted values therefore
belong in the script or in a file of yours next to it.

## Episode `.npz`

`<side>` is `left` or `right`.

| key | shape | meaning |
|---|---|---|
| `t_top` | [F] | top-camera frame times [s] |
| `t_left`, `t_right` | [F] | wrist-camera frame times [s] |
| `<camera>_state_index` | [F] int | optional: state row shown by each frame of `<camera>` (default `max(i - 1, 0)`) |
| `<side>_t` | [n] | arm-state times [s] (`left_t` is the canonical sample clock) |
| `<side>_q` | [n, 6] | measured joints [rad] |
| `<side>_qd` | [n, 7] | measured joint velocities [rad/s] (optional; column 6 is the gripper) |
| `<side>_tau` | [n, 7] | measured efforts [N·m] (optional; column 6 is the gripper motor, see Conventions) |
| `<side>_pose`, `<side>_cmd_pose` | [n, 4, 4] | logged end-effector poses (optional, informational) |
| `<side>_cmd_t`, `<side>_cmd` | [m], [m, 6] | joint commands [rad] at their timestamps [s] |
| `<side>_grip_t`, `<side>_grip` | [g], [g] | measured gripper opening [0..1] |
| `<side>_grip_cmd_t`, `<side>_grip_cmd` | [h], [h] | commanded gripper opening [0..1] |

**10 fps episodes** (`sib_1` .. `sib_4`) were built with
`episode_from_lerobot(timestamp, state, action, velocity=..., torque=..., time_offset=...)`.

- **Inputs.** LeRobot rows are `[left j1..j6, left grip, right j1..j6, right grip]`.
  Every row is a copy of the recorded sample nearest to its 10 fps tick.
- **Grid.** States, actions, velocities (`qd`) and efforts (`tau`) are linearly
  interpolated onto a 30 Hz grid that starts at the first tick. `time_offset`
  (the state stream's start minus the top camera's, from the LeRobot metadata)
  puts the grid on the clock of the 30 Hz recordings, seconds from the first top
  frame.
- **Frames.** `t_top`, `t_left` and `t_right` hold the tick times on that clock.
  Frame `k` shows state sample `3k + 2` (the last frame is clipped to the end of
  the grid), so `<camera>_state_index[k] = 3k + 2`: a 10 fps frame shows the state
  2/30 s *after* its tick (`frame_lag = -2/30`).

## Scene `.json`

`scenes/main.json`, abridged:

```json
{
 "name": "main", "episode": "episodes/main.npz",
 "table_z": 0.75,
 "bases": {"left": [0.2353, 0.3758, 0.75], "right": [0.2362, -0.2869, 0.75]},
 "bin": {"center_xy": [0.636, 0.132], "yaw_deg": 7.6, "top_size_m": [0.307, 0.219], "bottom_size_m": [0.297, 0.209],
         "height_m": 0.13, "wall_m": 0.004, "floor_m": 0.004},
 "objects": {
  "screwdriver": {
   "arm": "right", "shape": "screwdriver",
   "events": {"cmd_close": 22, "contact": 43, "liftoff": 47, "cmd_open": 83, "release": 84}, "open_cmd_row": 73,
   "start": {"pos": [0.5631, -0.2243, 0.7679], "quat_xyzw": [0.0021, 0.0500, -0.0417, 0.9979]},
   "grasp_start": {"pos": [0.5711, -0.2250, 0.7679], "quat_xyzw": [0.0021, 0.0500, -0.0417, 0.9979]},
   "grasp_height_m": 0.0179, "grasp_radius_m": 0.0127, "rest_pitch_rad": 0.10,
   "geometry": {"handle_profile": [[-0.057, 0.0135], [0.0, 0.0127], [0.053, 0.0105]], "shaft_radius_m": 0.003,
                "length_m": 0.210, "mass_kg": 0.07},
   "butt_local": [-0.057, 0, 0], "collar_local": [0.053, 0, 0], "tip_local": [0.153, 0, 0],
   "com_local": [0.035, 0, 0]
  }
 },
 "time_base": "top frame i shows arm-state sample i-1"
}
```

(The numbers above only illustrate the format; use the scene files.)

- **`bases`.** These are the arm base positions. Apply them with
  `set_arm_bases(builder, scene["bases"])` on a one-station builder before
  `add_world`.
- **`bin`.** This is an open box on the table, tapered and rotated about z.
  - **Pose.** `center_xy` is the centre of its footprint, and `yaw_deg` is the
    direction of its length (bin-frame x). The bin frame has its origin on the
    table under that centre (`bin_frame(bin, table_z)`).
  - **Size.** `bottom_size_m` and `top_size_m` are the outer length and width at
    the table and at the rim. `height_m` is the rim height above the table, and
    `wall_m` and `floor_m` are the wall and floor thicknesses.
  - **Outlines.** `bin_rim_polygon(bin)` gives the outer rim outline in world xy.
    `bin_inner_polygon(bin, height)` gives the inner wall at a height above the
    table, in bin-frame or world xy.
  - **Other keys.** `rim_corners_xy` and fit quality values are informational.
- **`events`.** These are top-frame indices of the real events, in the
  episode's own frame rate: the close command, finger contact, lift-off, the open
  command and release. The optional `event_rows` are the state rows they came
  from.
- **`open_cmd_row`.** This is the state row where the carry window of `score`
  ends: the last row before the logged gripper command (no delay) starts its
  rise to open the gripper, 0.02 above its hold value
  (`open_command_row(episode, obj)`).
- **`start`.** This is the rest pose of the screwdriver's body frame (see
  Conventions) before the gripper closes: `grasp_start` moved 8 mm toward the
  butt along the axis. The closing jaws of the real gripper push the screwdriver
  about 1 cm before they grip it (right wrist view of the main episode).
- **`grasp_start`.** This is the pose at the grasp, from
  `StationFK(station, scene["bases"]).object_starts(episode, scene)`; the
  ground-truth carry track starts from it.
  - **xy.** The midpoint of the grasping arm's two pad axes at the grasp height,
    averaged over the top frames `contact .. liftoff - 2`.
  - **z.** `table_z + grasp_height_m`, the height of the axis at the grasp point
    when the screwdriver rests on the table.
  - **Orientation.** The axis is perpendicular to the closing direction one frame
    before lift-off, on the side of `axis_hint_xy` (the tip direction), and
    pitched by `rest_pitch_rad` about body y (positive lowers the tip).
- **`geometry`.** The screwdriver is a body of revolution about its axis (body x).
  - **`handle_profile`.** These are `[x, radius]` stations [m] along the handle,
    from the butt to the collar. The radius is linear between stations.
  - **Shaft.** It runs from the last handle station to `tip_local`, with radius
    `shaft_radius_m`.
  - **Other values.** `length_m` is butt to tip. `mass_kg` is a nominal mass
    (assumed, not weighed).
  - **Radius lookup.** `handle_radius(obj, x)` evaluates this profile.
- **Points.** `butt_local`, `collar_local` (the front end of the handle),
  `tip_local` and `com_local` (the nominal centre of mass) are in the body frame.
  The metrics use `tip_local` and `com_local`, and `collar_local` for the wrist
  view.
- **`grasp_radius_m`.** This is the handle radius at the grasp point, the `r` of
  the holding test. Without it, `grasp_radius(obj)` takes the profile at x = 0.
- **Other fields.** `grip_gap_m` (measured jaw gap while held), `start_yaw_deg`,
  `frame_rate`, and `source` are informational.

**10 fps scenes** are derived from the logs and the recorded frames. Holds and their events come from the
gripper signals; events are 10 fps frame indices.

- **Ensembles.** `jitter_scene(scene, rng)` perturbs the screwdriver start the way
  the verification ensembles do: the grasp point moves in xy by N(0, 2 mm) per
  axis, clipped at 4 mm, and the screwdriver turns about the vertical through it
  by U(±5°).

## Ground truth `.npz` (`load_gt`)

| key | shape | meaning |
|---|---|---|
| `screwdriver_pos` | [F, 3] | grasp point per top frame [m]: `grasp_start` before contact, FK-attached to the hand from contact to release, NaN after |
| `screwdriver_src` | [F] int8 | optional: 2 hand-attached (FK), 4 resting start |
| `screwdriver_rest_xyz` | [3] | start grasp point [m] |
| `screwdriver_final_tip_xyz` | [3] | final tip position from the last top frames [m] |
| `screwdriver_final_yaw_deg` | scalar | final direction of the axis (butt to tip) in the xy plane [deg] |
| `screwdriver_final_xyz` | [3] | optional: final grasp point [m] |
| `t`, `frame`, `top_state_index` | [F] | optional: the top-frame clock |

- **Carry track.** `pos` comes from
  `StationFK(...).attached_tracks(episode, scene)`: the screwdriver carried
  rigidly by the hand (`<side>_link_6`) with the in-hand offset of the scene's
  `grasp_start` at the contact frame. The carry is not tracked in the images, because the
  gripper hides most of the screwdriver.
- **Final pose.** The final tip and yaw are measured in the last top frames, after
  the screwdriver has come to rest.

## `camera.json`

```json
{
 "top":   {"width": 848, "height": 480, "K": [9 values], "D": [k1, k2, p1, p2, k3],
           "distortion_model": "inverse_brown_conrady", "position": [x, y, z], "rotation_xyzw": [x, y, z, w]},
 "left":  {"width": 848, "height": 480, "K": [...], "D": [...], "distortion_model": "inverse_brown_conrady",
           "body": "left_camera_frame", "body_rotation_xyzw": [1, 0, 0, 0]},
 "right": {"...": "same as left with right_camera_frame"},
 "time_base": "top frame i shows arm-state sample i-1"
}
```

- **Top camera.** The pose is fitted to the recorded video. The camera looks
  along its −Z axis with +Y up.
- **Wrist cameras.** Each follows its MJCF body. The camera pose is
  `body_q[body] * (body_offset_m, body_rotation_xyzw)`, and `body_offset_m`
  defaults to zero. `wrist_camera_pose(recording, side, row, camera)` and
  `StationRenderer.wrist_pose` apply this mount.
- **Intrinsics.** These are the RealSense intrinsics of the recording. The
  inverse Brown-Conrady model maps distorted pixels to rays.

## Recording (one world, `Replay.recordings()`)

Rows are the episode's state samples `left_t` up to the current step. Values are
interpolated from the device history (recorded every `round(record_dt / dt)` steps,
default 1/120 s): linearly for positions and joints, and with normalized lerp for
rotations.

| key | shape | meaning |
|---|---|---|
| `t` | [n] | state times [s] |
| `top_row`, `<side>_wrist_row` | [F] | row shown by each top / wrist frame (`frame_rows`) |
| `screwdriver_pos`, `screwdriver_quat` | [n, 3], [n, 4] | body pose of the screwdriver (origin = grasp point) |
| `bin_pos`, `bin_quat` | [n, 3], [n, 4] | body pose of a movable bin (only if the world has a body `bin`) |
| `<side>_pad` | [n, 3] | midpoint of the two pads' grasp points (pad body point `(0, -0.0024, 0.071)`, 25 mm toward the fingertip from the pad box centre) |
| `<side>_hand_pos`, `<side>_hand_quat` | [n, 3], [n, 4] | `<side>_link_6` pose (frame of the in-hand metrics) |
| `<side>_camera_body_pos`, `<side>_camera_body_quat` | [n, 3], [n, 4] | `<side>_camera_frame` body pose (before the camera mount) |
| `<side>_finger` | [n] | left finger slide [m] |
| `<side>_q` | [n, 6] | simulated arm joints [rad] (`state.joint_q`; the solver must maintain it) |
| `<side>_q_real`, `<side>_grip_real` | [n, 6], [n] | measured joints and opening at the same rows |
| `body_q`, `body_index` | [n, B, 7], [B] | all body poses of the world and their global indices (for `recording_state` + rendering) |
| `world`, `complete` | scalars | world index; whether all state samples were simulated |

## `replay_common` API

**Data.**
- `load_episode(path_or_name)` loads an episode.
- `episode_from_lerobot(timestamp, state, action, velocity=None, torque=None, time_offset=0.098, rate=30, frame_lag=-2/30, cameras=("top", "left", "right"))`
  builds an episode from a LeRobot copy.
- `frame_rows(episode, camera="top")` gives the state row shown by each frame.
- `measured(episode, side)` returns the measured `(q, grip)` on `left_t`.
- `load_scene(path_or_name)` loads a scene and checks its screwdriver and bin entries.
- `load_gt(path_or_name)` loads ground truth.
- `load_camera(path=None)` loads camera.json.
- `jitter_scene(scene, rng, xy_sigma=0.002, xy_clip=0.004, yaw_deg=5)` perturbs the start.

**Geometry.**
- `set_arm_bases(builder, bases)` places the arm bases.
- `handle_radius(obj, x)` and `grasp_radius(obj)` give the screwdriver radius along its axis.
- `bin_frame(bin, table_z)`, `bin_rim_polygon(bin)` and `bin_inner_polygon(bin, height, world=False)` give
  the bin frame and outlines.
- `polygon_signed_distance(points_xy, polygon)` returns the signed distance to a
  convex polygon, negative inside.
- `quat_to_matrix(q)` converts quaternions to rotation matrices.
- `StationFK(station_xml, bases)` evaluates forward kinematics on the CPU:
  - `pose(arm_q, grip)` and `pose_row(episode, row)` return body poses.
  - `pads(body_q, side)` and `grasp_point(body_q, side)` return pad geometry.
  - `object_starts(episode, scene)` and `attached_tracks(episode, scene)` produce
    the grasp pose (`grasp_start`) and the FK-attached carry track.

**Commands.** `command_schedule(episode, times, delay)` returns
`{side: {"q": [T, 6], "finger": [T]}}`. `open_command_row(episode, obj)` gives
the scene's `open_cmd_row`.

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
  agree to a few micrometres over 0.5 s; the contact solver is not bitwise
  deterministic, so repeated runs differ slightly too.
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
  `{"complete", "arm_rmse_rad": {side}, "objects": {"screwdriver": metrics}, "all_held", "all_placed"}`.
- `check(metrics, thresholds=None)` returns `{"pass", "failed": [...]}` using
  `DEFAULT_THRESHOLDS`, the per-rollout counterparts of the verification gates.
- `summarize([metrics, ...])` returns, for an ensemble of one episode, the held
  and placed counts, `held_placed_rot_ok` (copies held, placed, and rotated at
  most `HELDOUT_ROT_MAX_DEG` = 12° in hand), and the medians of the metrics.

**Cameras.**
- `StationRenderer(model, camera=None, supersample=1, look=None)` provides:
  - `render_top(state, world=0, masks=False)`
  - `render_wrist(state, side, world=0, masks=False)`
  - `render(state, camera_name, pose, world, masks)`
  - `top_pose()` and `wrist_pose(state, side, world)`
- `render_top(model, state, world=0, camera=None, masks=False)` and
  `render_wrist(model, state, side, ...)` cache one renderer per model.
- Images are [480, 848, 3] uint8 sRGB (the size in camera.json). With
  `masks=True` the call also returns `{"screwdriver": bool [480, 848], "bin": bool [480, 848]}`
  visible pixels.
- `project_points(intrinsics, points, pose=None)` projects world points to pixels
  through the distortion model.
- `wrist_camera_pose(recording, side, row, camera)` returns the wrist camera pose
  of a recorded row, for `project_points`.
- `camera_rays(intrinsics, supersample)` returns the camera rays.

**Contacts.** `contact_summary(model, state, contacts, a, b=None, solver=None, world=None)`.
- **Selectors.** These are label substrings of a shape or its body (for example
  `"right_lf_down"`, `"screwdriver"`, `"table_plane"`, `"bin"`), shape indices, or
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

All metrics use state-sample rows. Event frames are converted with `top_row`.
The carry window runs from `row(liftoff) + 3` to the scene's `open_cmd_row` (at
most `row(release) - 2`), inclusive: it ends before the gripper command starts
to open, so it does not depend on the command delay.
The grasp point is the screwdriver's body origin, and `r` is `grasp_radius_m`.

| metric | definition |
|---|---|
| `held_fraction` (`held` ≥ 0.9) | carry rows with \|grasp point − pad midpoint of its arm\| ≤ r + 15 mm |
| `inhand_rot_deg` | largest rotation of the screwdriver in the grasping hand's frame (`<side>_link_6`) over the carry rows, relative to the first carry row |
| `slip_m` | largest drift of the grasp point in the hand frame over the carry rows |
| `placed` | complete episode; final tip and centre of mass (`tip_local`, `com_local`) inside the bin's inner wall polygon at their heights, dilated by 5 mm; centre of mass at least 2 cm below the rim; mean speed of the centre of mass under 2 cm/s over the last 0.3 s |
| `liftoff_err_rows` (`_s`) | first row after `row(cmd_close)` with the grasp point 1 cm above its start, minus `row(liftoff)` |
| `carry_track_err_m` | median xy distance to GT `screwdriver_pos` over carry rows (`carry_track_err_3d_m` in 3D, report only) |
| `grip_gap_err_mm` | median of 2 × (sim slide − measured opening × 0.0475), from `row(liftoff)` to `row(cmd_open)` |
| `final_tip_xy_err_m` | final tip to GT `screwdriver_final_tip_xyz`, in xy |
| `final_yaw_err_deg` | final axis yaw minus GT `screwdriver_final_yaw_deg`, wrapped to ±180° |
| `moved_before_grasp_m` | largest xy displacement of the grasp point from its start, up to `row(cmd_close)` |
| `max_rise_m` | highest grasp-point height above its start over the episode |
| `arm_rmse_rad` | per arm, over all rows |
| `release_err_s` (report) | first carry row with the grasp point 1 cm from its in-hand position at the carry start, minus `row(release)` |
| `wrist_collar_err_px`, `wrist_collar_v_range_px` (report) | the handle collar projected into the grasping arm's wrist camera on the frames that show carry rows: median distance to its tracked image position, and the range of its image row. They need `camera` and a tracked collar, which the workspace ground truth does not include, so they are `None` here. |

- **Placement details.** The placement test also reports `tip_wall_m` and
  `com_wall_m` (signed distances to the inner wall polygon, negative inside),
  `com_below_rim_m`, `final_speed_mps`, `final_tip_xyz`, `final_com_xyz`, and
  `in_bin` (the geometric part of `placed`). It also rejects a centre of mass
  more than 1 cm below the table.
- **Movable bin.** With a body `bin`, the bin frame of the test moves with that
  body from its first to its last recorded pose (`bin_moved_m`).
- **Per-rollout check.** `check()` compares one scored rollout with
  `DEFAULT_THRESHOLDS`, the single-rollout counterparts of the verification gates.
  Verification itself gates ensembles of jittered copies (counts of copies and
  medians over copies; see the task description).
- **10 fps episodes.** Their event frames are 100 ms apart, so the timing metrics
  are coarse there. On 10 fps episodes verification gates only per-copy outcomes
  and arm tracking (see the task description).
