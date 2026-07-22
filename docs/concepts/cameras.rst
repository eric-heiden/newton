.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Cameras
=======

Newton cameras are first-class model entities. Each camera is stored as a row in
several per-camera arrays on :class:`~newton.Model`, carries a projection
descriptor, and can be attached to a rigid body or fixed in the world frame.
Rays are generated on demand from the projection descriptor at render or sensor
evaluation time; no per-pixel data is stored on the model.

Camera space is **-Z forward, +Y up, +X right** throughout Newton. The
``camera_transform`` stores the camera-to-parent transform (parent is the
attached body when ``body >= 0``, or the world frame when ``body == -1``).

Overview
--------

Use :meth:`~newton.ModelBuilder.add_camera` to register cameras before calling
:meth:`~newton.ModelBuilder.finalize`:

.. code-block:: python

   import math
   import warp as wp
   import newton

   builder = newton.ModelBuilder()

   # World-fixed overview camera
   builder.add_camera(
       xform=wp.transform((0.0, 3.0, 1.5), wp.quat_identity()),
       projection=newton.CameraPinhole.from_fov(math.radians(60.0)),
       label="overview",
   )

   # Body-attached wrist camera
   body = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
   builder.add_camera(
       body=body,
       xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()),
       projection=newton.CameraPinhole.from_fov(math.radians(90.0)),
       label="wrist",
   )

   model = builder.finalize()

After finalization, cameras are accessible through these arrays on
:class:`~newton.Model`:

.. list-table::
   :header-rows: 1

   * - Attribute
     - Description
   * - ``camera_count``
     - Total number of cameras across all worlds.
   * - ``camera_label``
     - Python list of string labels, one per camera.
   * - ``camera_transform``
     - Camera-to-parent transforms, shape ``[camera_count]``, dtype ``wp.transform``.
   * - ``camera_body``
     - Body index each camera is attached to (``-1`` for world-fixed), shape ``[camera_count]``, dtype ``wp.int32``.
   * - ``camera_world``
     - World index of each camera (``-1`` when no worlds are configured), shape ``[camera_count]``, dtype ``wp.int32``.
   * - ``camera_flags``
     - Bitmask flags per camera (see :class:`~newton.CameraFlags`), shape ``[camera_count]``, dtype ``wp.int32``.
   * - ``camera_resolution``
     - Preferred resolution hint ``[width, height]`` in pixels, shape ``[camera_count, 2]``. ``[-1, -1]`` means no hint.
   * - ``camera_projection_index``
     - Index into ``camera_projections`` for each camera, shape ``[camera_count]``, dtype ``wp.int32``.
   * - ``camera_projections``
     - Deduplicated list of :class:`~newton.CameraProjection` descriptors.
   * - ``camera_world_start``
     - Per-world start index into the camera arrays, shape ``[world_count + 2]``, dtype ``wp.int32``.

World Transforms
----------------

:func:`~newton.eval_camera_world_xforms` computes the current world-space
camera-to-world transform for all cameras in one GPU kernel launch:

.. code-block:: python

   import newton

   # model and state from your simulation loop
   world_xforms = newton.eval_camera_world_xforms(model, state)
   # world_xforms: wp.array[wp.transform], shape [camera_count]

When ``state`` is ``None``, the model rest body transforms are used. Passing a
live :class:`~newton.State` is required for body-attached cameras to reflect
articulation motion.

Projection Descriptors
-----------------------

A projection descriptor describes how a camera maps scene geometry to pixels.
Descriptors are immutable value objects; parametric projections compare by
value, so :meth:`~newton.ModelBuilder.finalize` deduplicates them
automatically. Multiple cameras sharing the same :class:`~newton.CameraPinhole`
instance (or an equal one) will reference a single entry in
``model.camera_projections``.

All descriptor types inherit from :class:`~newton.CameraProjection`, which
provides the ``near`` and ``far`` clipping distance hints.

Pinhole
^^^^^^^

:class:`~newton.CameraPinhole` implements perspective projection in physical
(USD-style) form using ``focal_length``, ``horizontal_aperture``, and
``vertical_aperture``:

.. code-block:: python

   import math
   import newton

   # From a vertical field of view (most common)
   proj = newton.CameraPinhole.from_fov(math.radians(60.0), aspect=16.0 / 9.0)

   # Read the resolved FOV back
   print(proj.fov)   # vertical FOV in radians

   # Direct construction with USD physical parameters
   proj = newton.CameraPinhole(
       focal_length=24.0,        # aperture units (tenths of world unit by USD convention)
       horizontal_aperture=36.0,
       vertical_aperture=24.0,
   )

Fisheye — OpenCV
^^^^^^^^^^^^^^^^

:class:`~newton.CameraFisheyeOpenCV` uses the OpenCV model
``r = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)`` with
intrinsics in pixels:

.. code-block:: python

   import newton

   proj = newton.CameraFisheyeOpenCV(
       fx=400.0, fy=400.0,
       cx=320.0, cy=240.0,
       k1=-0.01, k2=0.0, k3=0.0, k4=0.0,
   )

When ``image_width`` / ``image_height`` are ``None``, the calibration space
equals the render resolution.

Fisheye — F-theta
^^^^^^^^^^^^^^^^^

:class:`~newton.CameraFisheyeFTheta` uses the polynomial
``r = k0 + k1*theta + k2*theta^2 + k3*theta^3 + k4*theta^4`` with optical
center in pixels:

.. code-block:: python

   import newton

   proj = newton.CameraFisheyeFTheta(
       optical_center_x=320.0,
       optical_center_y=240.0,
       k1=200.0,  # linear mapping (equidistant when k0=0, k1=f)
   )

Fisheye — Kannala-Brandt
^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~newton.CameraFisheyeKannalaBrandt` uses the K3 model
``r = k0*theta + k1*theta^3 + k2*theta^5 + k3*theta^7``:

.. code-block:: python

   import newton

   proj = newton.CameraFisheyeKannalaBrandt(
       optical_center_x=320.0,
       optical_center_y=240.0,
       k0=200.0,  # equidistant at default coefficients
   )

Custom Ray Bundles
^^^^^^^^^^^^^^^^^^

:class:`~newton.CameraCustomRays` wraps a precomputed Warp array of
camera-space rays (shape ``[height, width, 2]``, dtype ``wp.vec3f``; index 0
is the ray origin, index 1 is the normalized direction):

.. code-block:: python

   import warp as wp
   import newton

   rays = wp.zeros((480, 640, 2), dtype=wp.vec3f)
   # ... fill rays with your custom projection ...

   proj = newton.CameraCustomRays(rays=rays)

   # Pass the same instance to multiple cameras to share one ray bundle
   builder.add_camera(label="left", projection=proj)
   builder.add_camera(label="right", projection=proj)

``CameraCustomRays`` uses object identity for equality: two distinct instances
are never considered equal even if their ray arrays contain identical data. Pass
the same Python object to multiple cameras to share one ray bundle in memory.
The ``rays`` array determines the fixed resolution; passing any other ``(width,
height)`` pair to ``compute_camera_rays`` raises a ``ValueError``.

Resolution Hint vs. Renderer Override
--------------------------------------

The ``resolution`` argument to :meth:`~newton.ModelBuilder.add_camera` (and the
matching ``camera_resolution`` array on :class:`~newton.Model`) is a *hint*. A
renderer or sensor that has its own resolution (for example a tiled camera with
a fixed output buffer) will use its own resolution and ignore the hint.
``[-1, -1]`` means no hint was provided.

USD Import
----------

:meth:`~newton.ModelBuilder.add_usd` imports ``UsdGeom.Camera`` prims as model
cameras when ``load_cameras=True`` (the default):

.. code-block:: python

   import newton

   builder = newton.ModelBuilder()
   result = builder.add_usd("scene.usda", load_cameras=True)

   path_camera_map = result["path_camera_map"]
   # Maps prim path (str) -> camera index in the builder

Camera prims parented under a ``UsdPhysics.RigidBodyAPI`` prim are attached to
the corresponding body; all others are world-fixed. Focal length, aperture, and
clipping range are read from the standard USD attributes and mapped to
:class:`~newton.CameraPinhole`. Orthographic cameras are skipped with a
warning.

MJCF Import
-----------

:meth:`~newton.ModelBuilder.add_mjcf` imports ``<camera>`` elements:

.. code-block:: python

   import newton

   builder = newton.ModelBuilder()
   builder.add_mjcf("robot.xml")

Each ``<camera>`` element maps to one model camera:

- ``fovy`` (degrees) is converted to the vertical FOV of a
  :class:`~newton.CameraPinhole`.
- If both ``focal`` and ``sensorsize`` are present they are used directly as
  the physical aperture parameters.
- ``resolution`` is stored as the resolution hint.
- Cameras inside a ``<body>`` are attached to that body; cameras inside a
  ``<frame>`` inherit the frame's composed transform.

Newton does not support the MJCF runtime tracking modes (``trackbody``,
``trackcom``, ``targetbody``, ``targetbodycom``). Cameras using any non-fixed
``mode`` are imported as fixed cameras; the original mode string is preserved
on the model as the custom attribute ``mjcf:camera_mode`` so downstream code
can apply tracking logic.

Camera Flags
------------

:class:`~newton.CameraFlags` is a bitmask stored per camera in
``model.camera_flags``:

- ``CameraFlags.ENABLED`` (1) — the camera is active.
- ``CameraFlags.VISIBLE`` (2) — the camera frustum is shown in the viewer when
  ``show_cameras`` is enabled.

Both flags are set by default. Pass ``enabled=False`` to
:meth:`~newton.ModelBuilder.add_camera` to create a disabled camera.

Viewer Integration
------------------

Interactive viewers expose two camera-related options per layer:

- ``viewer.show_cameras = True`` — draws a frustum wireframe for every camera
  in the model.
- ``viewer.camera_frustum_depth = 0.5`` — controls the depth of the frustum
  wireframe in meters.

:meth:`~newton.viewer.ViewerBase.set_camera_from_model` snaps the viewport
camera to a model camera's current world pose:

.. code-block:: python

   # Snap viewport to the camera labelled "overview" using the current state
   viewer.set_camera_from_model("overview", state=state)

   # Or by index
   viewer.set_camera_from_model(0)

The ViewerGL GUI additionally provides a **Camera** dropdown that lists all
model cameras by label, with a **Follow** checkbox that continuously tracks the
selected camera each frame.

Sensor-side Consumption
-----------------------

Per-world ray bundles, lazy projection-keyed caching, and the
``SensorBatchedCamera`` integration that feeds Newton cameras into a rendering
pipeline ship as a separate plan after ``SensorBatchedCamera`` lands on this
branch. The model-side schema (``Model.camera_*`` arrays,
:func:`~newton.eval_camera_world_xforms`, descriptor equality/dedup) is stable
and forms the interface guarantee for that work.

See Also
--------

* :doc:`sensors` -- available sensor types and the sensor update pattern
* :doc:`../api/newton` -- ``newton`` top-level API reference
  (``CameraProjection``, ``CameraPinhole``, ``CameraFisheyeOpenCV``,
  ``CameraFisheyeFTheta``, ``CameraFisheyeKannalaBrandt``,
  ``CameraCustomRays``, ``CameraFlags``, ``eval_camera_world_xforms``)
* :meth:`~newton.ModelBuilder.add_camera` -- full parameter reference
