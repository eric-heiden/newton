.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Sensors
========

Sensors in Newton provide a way to extract measurements and observations from the simulation. They compute derived
quantities that are commonly needed for control, reinforcement learning, robotics applications, and analysis.

Overview
--------

Most Newton sensors follow a common pattern:

1. **Initialization**: Configure the sensor with the model and specify what to measure
2. **Update**: Call ``sensor.update(state, ...)`` during the simulation loop to compute measurements
3. **Access**: Read results from sensor attributes (typically as Warp arrays)

.. note::

   Sensors automatically request any :doc:`extended attributes <extended_attributes>` they need
   (e.g. ``body_qdd``, ``Contacts.force``) at init, so ``State`` and ``Contacts`` objects created afterwards will
   include them.

   ``SensorContact`` additionally requires a call to ``solver.update_contacts()`` before ``sensor.update()``.

   ``SensorCamera`` writes results to output arrays passed into ``update()`` rather than storing them as sensor
   attributes.

.. testcode::

   import warp as wp
   import newton
   from newton.sensors import SensorIMU

   # Build the model
   builder = newton.ModelBuilder()
   builder.add_ground_plane()
   body = builder.add_body(xform=wp.transform((0, 0, 1), wp.quat_identity()))
   builder.add_shape_sphere(body, radius=0.1)
   builder.add_site(body, label="imu_0")
   model = builder.finalize()

   # 1. Create sensor and specify what to measure
   imu = SensorIMU(model, sites="imu_*")

   # Create solver and state
   solver = newton.solvers.SolverMuJoCo(model)
   state = model.state()

   # Simulation loop
   for _ in range(100):
       state.clear_forces()
       solver.step(state, state, None, None, dt=1.0 / 60.0)

       # 2. Compute measurements from the current state
       imu.update(state)

       # 3. Results stored on sensor attributes
       acc = imu.accelerometer.numpy()   # (n_sensors, 3) linear acceleration
       gyro = imu.gyroscope.numpy()      # (n_sensors, 3) angular velocity

   print("accelerometer shape:", acc.shape)
   print("gyroscope shape:", gyro.shape)

.. testoutput::

   accelerometer shape: (1, 3)
   gyroscope shape: (1, 3)

.. _label-matching:

Label Matching
--------------

Several Newton APIs accept **label patterns** to select bodies, shapes, joints, sites, etc. by name. Parameters that
support label matching accept one of the following:

- A **list of integer indices** -- selects directly by index.
- A **single string pattern** -- selects all entries whose label matches the pattern via :func:`fnmatch.fnmatch`
  (supports ``*`` and ``?`` wildcards).
- A **list of string patterns** -- selects all entries whose label matches at least one pattern.
- A **compiled string regular expression** -- selects all entries whose entire label or name matches the expression via
  :meth:`re.Pattern.fullmatch`.

Ordinary strings always use glob syntax. Compile a pattern with :func:`re.compile` to opt into regular-expression
syntax. Callers who want a regular expression to match a substring can add ``.*`` around that substring explicitly.
For :class:`~newton.selection.ArticulationView`, ``pattern`` is matched against full articulation labels. Joint and
link filters are matched against the final path component of each label. :meth:`~newton.Model.find_bodies`,
:meth:`~newton.Model.find_shapes`, :meth:`~newton.Model.find_joints`, :meth:`~newton.Model.find_joint_dofs`, and
:meth:`~newton.Model.find_joint_coords` return the model indices a pattern selects, matching full labels and final path
components, optionally within one world.
:class:`~newton.selection.WorldView` matches labels the same way and selects joint DOFs and coordinates by the labels
of their joints.

.. code-block:: python

   import re

   # single pattern: all shapes whose label starts with "foot_"
   SensorIMU(model, sites="foot_*")

   # compiled regular expression: full-match an environment and object label
   SensorIMU(model, sites=re.compile(r"/World/envs/env_[0-9]+/imu_(left|right)"))

   # list of patterns: union of two groups
   SensorContact(model, sensing_shapes=["*Plate*", "*Flap*"])

   # list of indices: explicit selection
   SensorFrameTransform(model, shapes=[0, 3, 7], reference_sites=[1])

Available Sensors
-----------------

Newton provides five sensor types. See the
:doc:`API reference <../api/newton_sensors>` for constructor arguments,
attributes, and usage examples.

* :class:`~newton.sensors.SensorContact` -- contact forces between bodies or shapes, with friction decomposition,
  optional per-counterpart force matrices, and force-weighted contact positions.
* :class:`~newton.sensors.SensorFrameTransform` -- relative transforms of shapes/sites with respect to reference sites.
* :class:`~newton.sensors.SensorIMU` -- linear acceleration and angular velocity at site frames.
* :class:`~newton.sensors.SensorCamera` -- raytraced color, HDR color, depth, forward-depth, normal, albedo, and
  shape-index rendering; one view per camera transform, mapped to worlds via a per-view selector.
* :class:`~newton.sensors.SensorTiledCamera` -- deprecated; superseded by :class:`~newton.sensors.SensorCamera`.

Camera Rays from USD and Calibration Data
-----------------------------------------

:class:`~newton.sensors.SensorCamera` renders one view per world-space camera transform passed to
:meth:`~newton.sensors.SensorCamera.update`. The caller owns the camera-space rays and the per-view transforms.
The ray bundle for a standard USD pinhole camera can be built directly with
:meth:`~newton.sensors.SensorCamera.compute_camera_rays_usd_pinhole`, and the matching world-space per-view
transforms with :meth:`~newton.sensors.SensorCamera.compute_camera_transforms_usd` (which converts the USD stage's
up axis to the model's and composes an optional import ``xform``). For lens models without standard USD attributes,
read the attributes you use in your pipeline and pass the numeric values into the matching helper:

.. code-block:: python

   from pxr import Usd

   from newton.sensors import SensorCamera

   stage = Usd.Stage.Open("scene.usda")
   usd_camera = stage.GetPrimAtPath("/World/Camera")

   camera = SensorCamera(model)
   camera.create_default_light()

   # Camera-space rays for one 640x480 pinhole camera, on the model device.
   camera_rays = SensorCamera.compute_camera_rays_usd_pinhole(640, 480, usd_camera, device=model.device)

   # World-space transform per view, read from the USD camera(s).
   camera_transforms = camera.compute_camera_transforms_usd(usd_camera)
   view_count = camera_transforms.shape[0]

   color = camera.create_color_image_output(view_count, 640, 480)

   # update() syncs deformable-mesh points from state by default; refit the
   # shape/particle BVHs first on any frame whose geometry moved.
   camera.update(state, camera_transforms, camera_rays, color_image=color)

For OpenCV-calibrated pinhole cameras, call
:meth:`~newton.sensors.SensorCamera.compute_camera_rays_pinhole_opencv` with the calibrated intrinsics and
radial, tangential, and optional thin-prism coefficients.

For fisheye cameras, extract the calibration values from your chosen USD attributes and call one of
:meth:`~newton.sensors.SensorCamera.compute_camera_rays_fisheye_opencv`,
:meth:`~newton.sensors.SensorCamera.compute_camera_rays_fisheye_ftheta`, or
:meth:`~newton.sensors.SensorCamera.compute_camera_rays_fisheye_kannala_brandt`. Each helper builds a single-camera
``(height, width, 2)`` ray bundle.

Calibrated Camera Geometry
--------------------------

:class:`SensorCamera.Intrinsics <newton.sensors.SensorCamera.Intrinsics>` describes a calibrated pinhole camera:
image size, focal lengths, principal point, and either OpenCV distortion (radial ``k1``-``k6``, tangential ``p1``,
``p2``, thin-prism ``s1``-``s4``) or the RealSense inverse Brown-Conrady model, whose polynomial maps recorded
(distorted) pixels to rays. It converts between world points and image coordinates on the host in float64 NumPy and
builds the ray bundle that renders through the same lens:

* :meth:`~newton.sensors.SensorCamera.Intrinsics.from_camera_matrix`,
  :meth:`~newton.sensors.SensorCamera.Intrinsics.from_dict`, and
  :meth:`~newton.sensors.SensorCamera.Intrinsics.from_json` -- intrinsics from a camera matrix and coefficients, or
  from a calibration dictionary or JSON file (OpenCV-style ``K``, ``D``, and ``distortion_model``, ROS
  ``CameraInfo`` and calibration files, RealSense ``rs2_intrinsics``);
* :meth:`~newton.sensors.SensorCamera.Intrinsics.project` -- world points [m] to image coordinates [px] and forward
  depth [m];
* :meth:`~newton.sensors.SensorCamera.Intrinsics.unproject` -- image coordinates to world-space unit ray directions
  from the camera position;
* :meth:`~newton.sensors.SensorCamera.Intrinsics.unproject_to_depth` -- image coordinates and forward depths to world
  points;
* :meth:`~newton.sensors.SensorCamera.Intrinsics.unproject_to_plane` -- image coordinates to the points where their
  rays meet a plane ``a*x + b*y + c*z + d = 0``;
* :meth:`~newton.sensors.SensorCamera.Intrinsics.resize` -- the same camera for a resampled image;
* :meth:`~newton.sensors.SensorCamera.Intrinsics.compute_camera_rays` -- the camera-space ray bundle for
  :meth:`~newton.sensors.SensorCamera.update`.

Image coordinates follow OpenCV: x right, y down, and the center of the top-left pixel at ``(0, 0)``, so pixel
``(i, j)`` of an image array (column ``i``, row ``j``) is centered at ``(i, j)``. Results that do not exist, such as
points behind the camera, points past the radius where a distortion polynomial folds back, or rays that miss a plane,
are NaN.

Camera transforms use the :class:`~newton.sensors.SensorCamera` frame: the camera looks along its local -Z axis with
+Y up. An OpenCV or ROS optical frame (+Z forward, +Y down) is that frame rotated by 180 degrees about X. For cameras
mounted on bodies, :meth:`~newton.sensors.SensorCamera.compute_camera_transforms_body` composes each body pose in a
state with the camera pose in the body frame:

.. code-block:: python

   import numpy as np
   import warp as wp

   from newton.sensors import SensorCamera

   # A RealSense calibration: camera matrix K and coefficients (k1, k2, p1, p2, k3) at 640x480.
   intrinsics = SensorCamera.Intrinsics.from_camera_matrix(
       K, D, width=640, height=480, distortion_model="inverse_brown_conrady"
   )

   # A fixed camera: position [m] and xyzw quaternion of the camera frame in the world.
   top = wp.transform(wp.vec3(0.05, 0.02, 1.77), wp.quat(-0.19, 0.17, 0.68, -0.68))
   pixels, forward_depth = intrinsics.project(points, top)
   on_table = intrinsics.unproject_to_plane(pixels, top, plane=(0.0, 0.0, 1.0, -0.75))

   # A wrist camera whose optical frame sits at `offset` in the frame of body `wrist`.
   mount = wp.transform(offset.p, offset.q * wp.quat(1.0, 0.0, 0.0, 0.0))
   wrist_transforms = SensorCamera.compute_camera_transforms_body(state.body_q, [wrist], [mount])
   wrist_pixels, _ = intrinsics.project(points, wrist_transforms.numpy()[0])

   # Render through the same lens; points rendered at a pixel project to its center.
   camera = SensorCamera(model)
   camera.create_default_light()
   rays = intrinsics.compute_camera_rays(device=model.device)
   color = camera.create_color_image_output(1, intrinsics.width, intrinsics.height)
   camera.update(state, wrist_transforms, rays, color_image=color)

:meth:`~newton.sensors.SensorCamera.compute_camera_rays_pinhole_opencv` samples pixel ``(i, j)`` at calibration image
coordinates ``(i + 0.5, j + 0.5)`` (scaled to the output size), half a pixel from the OpenCV pixel centers that
:class:`SensorCamera.Intrinsics <newton.sensors.SensorCamera.Intrinsics>` uses.

From pixels in a photo to points on a plane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~newton.sensors.SensorCamera.Intrinsics.from_dict` and
:meth:`~newton.sensors.SensorCamera.Intrinsics.from_json` read the intrinsics of one camera from a calibration file
and ignore other keys, such as the camera pose. This example reads a calibration with an inverse Brown-Conrady lens,
places the camera from the pose stored next to its intrinsics, finds where the rays through pixels of a photo meet a
table top, and projects those points back to the same pixels:

.. testcode:: calibration-file

   import json

   import numpy as np
   import warp as wp

   from newton.sensors import SensorCamera

   # A calibration file with one entry per camera; "overhead" looks down at a table.
   calibration = json.loads("""
   {"overhead": {"width": 640, "height": 480,
                 "K": [600.0, 0.0, 319.5, 0.0, 600.0, 239.5, 0.0, 0.0, 1.0],
                 "D": [0.05, -0.02, 0.001, -0.0005, 0.0],
                 "distortion_model": "inverse_brown_conrady",
                 "position": [0.1, 0.0, 1.6], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}}
   """)
   intrinsics = SensorCamera.Intrinsics.from_dict(calibration, camera="overhead")
   # From the file itself: SensorCamera.Intrinsics.from_json("camera.json", camera="overhead")

   # The pose of the camera frame in the world (-Z forward, +Y up). For an OpenCV optical
   # frame (+Z forward, +Y down), multiply its rotation by wp.quat(1.0, 0.0, 0.0, 0.0).
   entry = calibration["overhead"]
   pose = wp.transform(wp.vec3(*entry["position"]), wp.quat(*entry["rotation_xyzw"]))

   # Pixels of an object in a photo, e.g. the centroid of a segmentation mask: columns are x, rows are y.
   mask = np.zeros((480, 640), dtype=bool)
   mask[200:260, 300:380] = True
   rows, columns = np.nonzero(mask)
   pixels = np.array([[columns.mean(), rows.mean()], [40.0, 30.0]])

   # The points [m] where the rays through these pixels meet the table top z = 0.75 m.
   on_table = intrinsics.unproject_to_plane(pixels, pose, plane=(0.0, 0.0, 1.0, -0.75))
   print(on_table.round(3).tolist())

   # Projecting the points gives the pixels again.
   back, forward_depth = intrinsics.project(on_table, pose)
   print(np.allclose(back, pixels, atol=1e-6))

.. testoutput:: calibration-file

   [[0.128, 0.014, 0.75], [-0.302, 0.301, 0.75]]
   True

Camera Lighting
---------------

:meth:`~newton.sensors.SensorCamera.create_default_light` adds one directional light that shines down at an angle
relative to the model's up axis; ``direction`` places it and a linear RGB ``color`` sets its intensity and tint.
:meth:`~newton.sensors.SensorCamera.set_ambient_light` sets the hemispheric ambient light: surfaces facing along the
up axis receive the sky color, surfaces facing away from it the ground color, and other orientations a linear blend.

Extended Attributes
-------------------

Some sensors depend on extended attributes that are not allocated by default:

- ``SensorIMU`` requires ``State.body_qdd`` (rigid-body accelerations). By
  default it requests this from the model at construction, so subsequent
  ``model.state()`` calls allocate it automatically. Both
  :class:`~newton.solvers.SolverKamino` and
  :class:`~newton.solvers.SolverMuJoCo` populate this attribute. Kamino reports
  the discrete step-average center-of-mass acceleration in the world frame;
  impact steps therefore include the velocity impulse divided by the step
  duration.
- ``SensorContact`` requires ``Contacts.force`` (per-contact spatial force
  wrenches). By default it requests this from the model at construction, so
  subsequent :meth:`CollisionPipeline.contacts <newton.CollisionPipeline.contacts>` calls allocate it automatically. The solver
  must also support populating contact forces.

Performance Considerations
--------------------------

Sensors are designed to be efficient and GPU-friendly, computing results in
parallel where possible. Create each sensor once during setup and reuse it
every step -- this lets Newton pre-allocate output arrays and avoid per-frame
overhead.

Sensors that depend on extended attributes (e.g. ``body_qdd``,
``Contacts.force``) may add nontrivial cost to the solver step itself, since
the solver must compute and store these additional quantities regardless of
whether the sensor is evaluated after each step.

See Also
--------

* :doc:`sites` -- using sites as sensor attachment points and reference frames
* :doc:`../api/newton_sensors` -- full sensor API reference
* :doc:`extended_attributes` -- optional ``State``/``Contacts`` arrays required by some sensors
* ``newton.examples.sensors.example_sensor_contact`` -- SensorContact example
* ``newton.examples.sensors.example_sensor_imu`` -- SensorIMU example
* ``newton.examples.sensors.example_sensor_camera`` -- SensorCamera example (run with ``python -m newton.examples sensor_camera``)
