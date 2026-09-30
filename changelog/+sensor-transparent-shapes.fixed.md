Fix fully transparent shapes (`Model.shape_opacity` 0, e.g. MJCF geoms with rgba alpha 0) rendering as opaque black in `SensorTiledCamera`; they are now left out of the shape BVH.
