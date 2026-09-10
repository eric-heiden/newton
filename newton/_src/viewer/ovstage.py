# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional ovstage pose transport. No physics or appearance parsing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

if TYPE_CHECKING:
    import newton


@wp.kernel(enable_backward=False)
def _body_matrices(
    body_q: wp.array[wp.transform],
    indices: wp.array[int],
    offsets: wp.array[wp.mat44d],
    output: wp.array[wp.mat44d],
):
    i = wp.tid()
    pose = body_q[indices[i]]
    p = wp.vec3d(wp.float64(pose[0]), wp.float64(pose[1]), wp.float64(pose[2]))
    q = wp.quatd(wp.float64(pose[3]), wp.float64(pose[4]), wp.float64(pose[5]), wp.float64(pose[6]))
    output[i] = offsets[i] * wp.transpose(wp.transform_compose(p, q, wp.vec3d(1.0)))


class OvstageBodyBinding:
    """Publish Newton rigid-body poses into an existing ovstage.

    .. experimental::
        This type is experimental and may change without prior notice.

    The application owns the stage, model, publication floor and renderer.
    This binding never imports USD, creates geometry, or advances the floor.
    ``body_local_transforms`` uses USD's row-vector convention: each output is
    ``body_local_transforms[i] @ body_world[body_indices[i]]``. The full affine
    offset preserves scale, reflection and the residual transform after fixed
    joint collapse. Multiple prims may follow one body. Paths must identify
    existing, writable xformable prims, not instance-prototype children.

    Supply indices for the *final* model. Recreate the binding after topology,
    replication or mapping changes. Units and world placement must already
    agree. This prototype does not bind deformable vertices or joint state.

    Calls are synchronous with respect to payload writes. Complete rendering
    before the next write or close: ordinals do not provide historical stage
    snapshots. Do not destroy the stage while this binding is alive.
    """

    def __init__(
        self,
        stage: Any,
        model: newton.Model,
        *,
        ordinal: int,
        prim_paths: Sequence[str],
        body_indices: Sequence[int],
        body_local_transforms: np.ndarray,
    ):
        """Bind explicit paths to final-model indices and affine rest offsets.

        Args:
            stage: Caller-owned ovstage 0.2 Stage.
            model: Newton model whose State arrays will drive the prims.
            ordinal: Committed ordinal at which to validate destination prims.
            prim_paths: Unique absolute prim paths, in payload order.
            body_indices: One final-model body index per path.
            body_local_transforms: Float64 array of shape ``(N, 4, 4)``.

        Raises:
            ValueError: If mapping dimensions, indices or transforms are invalid.
            ImportError: If the optional ovstage package is unavailable.
        """
        paths = tuple(prim_paths)
        indices = np.asarray(body_indices)
        offsets = np.asarray(body_local_transforms, dtype=np.float64)
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("ordinal must be a nonnegative committed ordinal")
        if not paths or any(not isinstance(p, str) or not p.startswith("/") or p == "/" for p in paths):
            raise ValueError("prim_paths must contain absolute prim paths")
        if len(set(paths)) != len(paths):
            raise ValueError("Each destination prim must occur exactly once")
        if indices.shape != (len(paths),) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("body_indices must contain one integer per prim path")
        if np.any(indices < 0) or np.any(indices >= model.body_count):
            raise ValueError("body_indices must refer to the final model")
        if offsets.shape != (len(paths), 4, 4) or not np.isfinite(offsets).all():
            raise ValueError("body_local_transforms must be finite with shape (N, 4, 4)")
        if not np.allclose(offsets[:, :, 3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("body_local_transforms must be affine row-vector matrices")

        import ovstage  # noqa: PLC0415 - optional backend, loaded only when binding

        self.stage = stage
        self.model = model
        self.prim_paths = paths
        self._indices = wp.array(indices, dtype=int, device=model.device)
        self._offsets = wp.array(offsets, dtype=wp.mat44d, device=model.device)
        self._matrices = wp.empty(len(paths), dtype=wp.mat44d, device=model.device)
        self._event = wp.Event(model.device) if model.device.is_cuda else None
        self._reset = np.ones(len(paths), dtype=np.bool_)
        self._ordinal = ordinal
        self._closed = False
        self._dictionary = ovstage.PathDictionary(stage)
        self._paths = None
        try:
            self._paths = self._dictionary.create_path_list_from_strings(list(paths))
            self._query = stage.query_from_path_list(self._paths)
            found = set()
            token = self._dictionary.intern_token("omni:xform")
            with stage.read_attributes(self._query, [token], ovstage.OrdinalRange.latest(ordinal)) as read:
                read.wait()
                group = read.fetch_next()
                while group is not None:
                    try:
                        # A read group is emitted only for a present column.
                        found.update(group.prim_index(i) for i in range(group.prim_count))
                    finally:
                        stage.release_group(group)
                    group = read.fetch_next()
            if found != set(range(len(paths))):
                missing = [path for i, path in enumerate(paths) if i not in found]
                raise ValueError(f"Destination prims have no populated omni:xform: {missing}")
        except Exception:
            if hasattr(self, "_query"):
                self._query.release().wait()
            if self._paths is not None:
                self._dictionary.destroy_path_list(self._paths)
            self._dictionary.destroy()
            raise

    def write(self, state: newton.State, *, ordinal: int) -> None:
        """Write poses at an uncommitted ordinal, without publishing the stage.

        Args:
            state: State with body poses matching this binding's model/device.
            ordinal: Strictly increasing, nonnegative application ordinal.

        The caller then completes other scene writes, advances the global
        write floor, and renders that ordinal before calling this again.
        """
        if self._closed:
            raise RuntimeError("The ovstage binding is closed")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0 or ordinal <= self._ordinal:
            raise ValueError("ordinal must be a strictly increasing nonnegative integer")
        if (
            state.body_q is None
            or state.body_q.shape != (self.model.body_count,)
            or state.body_q.dtype != wp.transform
            or state.body_q.device != self.model.device
        ):
            raise ValueError("State body poses must match the binding's model and device")

        import ovstage  # noqa: PLC0415 - optional backend

        wp.launch(
            _body_matrices,
            dim=len(self.prim_paths),
            inputs=[state.body_q, self._indices, self._offsets, self._matrices],
            device=self.model.device,
        )
        if self._event is not None:
            wp.record_event(self._event)
        # Reserve the ordinal even if a later stage operation fails: writes
        # are not transactions and the caller must recover with a new ordinal.
        self._ordinal = ordinal
        self.stage.write_attribute(self._query, "omni:resetXformStack", ordinal, self._reset, is_array=False).wait()
        self.stage.write_attribute(
            self._query,
            "omni:xform",
            ordinal,
            self._matrices,
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
            cuda_event=self._event.cuda_event if self._event is not None else None,
        ).wait()

    def close(self) -> None:
        """Release binding queries; leave the caller-owned stage alive."""
        if not self._closed:
            self._query.release().wait()
            self._dictionary.destroy_path_list(self._paths)
            self._dictionary.destroy()
            self._closed = True

    def __enter__(self) -> OvstageBodyBinding:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
