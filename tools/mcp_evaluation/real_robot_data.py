# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Prepare measured Panda references and remove authored dynamics from its asset.

The optional pandas dependency is used only to convert the pinned publisher's
DataFrames. Runtime references contain numeric NPZ arrays and require NumPy only.
Derived measured data retain the publisher's CC-BY-SA-4.0 data license.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import pickle
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import ClassVar

import numpy as np

DATASET_DOI = "10.5281/zenodo.12516500"
DATASET_URL = "https://zenodo.org/api/records/12516500/files/PUB-5510-LIP4RID.zip/content"
DATASET_ARCHIVE_MD5 = "d3e29fbd280fc4cf08d008eb50559338"
DATASET_LICENSE = "CC-BY-SA-4.0"
TRAINING_SEEDS = (2, 3, 4)
HELDOUT_SEEDS = (21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38)
REFERENCE_KEYS = {"time", "q", "qd", "qdd", "tau", "episode_offsets", "episode_ids"}


def file_digest(path: Path, *, algorithm: str = "sha256") -> str:
    """Hash a file without loading the full source archive into memory."""
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class _DataFrameUnpickler(pickle.Unpickler):
    """Admit only the constructors needed by the publisher's numeric DataFrames."""

    _allowed: ClassVar[set[tuple[str, str]]] = {
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.internals.managers", "BlockManager"),
        ("pandas.core.internals.blocks", "new_block"),
        ("functools", "partial"),
        ("pandas._libs.internals", "_unpickle_block"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("builtins", "slice"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self._allowed:
            raise pickle.UnpicklingError(f"Unsupported DataFrame constructor: {module}.{name}")
        if (module, name) == ("pandas.core.internals.blocks", "new_block"):
            module, name = "pandas._libs.internals", "_unpickle_block"
        return super().find_class(module, name)


def load_reference(path: Path) -> dict[str, np.ndarray]:
    """Load finite measured episodes with positions [rad], times [s], and torques [N·m]."""
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != REFERENCE_KEYS:
            raise ValueError(f"Reference fields must be exactly {sorted(REFERENCE_KEYS)}")
        data = {key: np.asarray(source[key]).copy() for key in source.files}
    n = len(data["time"])
    if data["time"].shape != (n,) or any(data[key].shape != (n, 7) for key in ("q", "qd", "qdd", "tau")):
        raise ValueError("Reference time must be [n]; joint observations must be [n, 7]")
    if not all(np.isfinite(value).all() for value in data.values()):
        raise ValueError("Reference arrays must be finite")
    offsets, ids = data["episode_offsets"], data["episode_ids"]
    if offsets.dtype.kind not in "iu" or ids.dtype.kind not in "iu":
        raise ValueError("Episode offsets and identifiers must be integer arrays")
    if offsets.ndim != 1 or ids.ndim != 1 or len(offsets) != len(ids) + 1:
        raise ValueError("Episode offsets must have one more entry than episode identifiers")
    if not len(ids) or len(np.unique(ids)) != len(ids) or offsets[0] != 0 or offsets[-1] != n:
        raise ValueError("Episode identifiers and boundaries must be complete and unique")
    for start, stop in itertools.pairwise(offsets):
        times = data["time"][start:stop]
        if stop - start < 100 or not np.all(np.diff(times) > 0) or times[-1] - times[0] < 2:
            raise ValueError("Each episode must contain increasing timestamps and sufficient recorded motion")
    return data


def prepare_reference(archive_path: Path, output_path: Path, *, split: str) -> dict:
    """Convert only measured training or held-out Panda fields from the pinned archive."""
    start = time.perf_counter()
    if split not in ("training", "heldout"):
        raise ValueError("split must be training or heldout")
    if file_digest(archive_path, algorithm="md5") != DATASET_ARCHIVE_MD5:
        raise ValueError("Archive does not match the publisher's versioned checksum")
    seeds, sinusoids = (TRAINING_SEEDS, 50) if split == "training" else (HELDOUT_SEEDS, 100)
    pieces = {key: [] for key in ("time", "q", "qd", "qdd", "tau")}
    offsets, records = [0], []
    with zipfile.ZipFile(archive_path) as archive:
        for seed in seeds:
            name = f"Data/Robots/PANDA/Experiments/panda7dof_num_sin_{sinusoids}_seed_{seed}_as_filtered_fcut4.0.pkl"
            raw = archive.read(name)
            frame = _DataFrameUnpickler(io.BytesIO(raw)).load()
            for key, prefix in (("q", "q"), ("qd", "dq"), ("qdd", "ddq"), ("tau", "tau_interp")):
                pieces[key].append(frame[[f"{prefix}_{joint}" for joint in range(1, 8)]].to_numpy(dtype=np.float64))
            times = frame["t"].to_numpy(dtype=np.float64)
            pieces["time"].append(times)
            offsets.append(offsets[-1] + len(times))
            records.append(
                {
                    "seed": seed,
                    "archive_member": name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "samples": len(times),
                    "time_first_s": float(times[0]),
                    "time_last_s": float(times[-1]),
                    "interval_quantiles_s": np.quantile(np.diff(times), [0, 0.01, 0.5, 0.99, 1]).tolist(),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.concatenate(value) for key, value in pieces.items()}
    np.savez_compressed(
        output_path,
        **arrays,
        episode_offsets=np.asarray(offsets, dtype=np.int64),
        episode_ids=np.asarray(seeds, dtype=np.int64),
    )
    load_reference(output_path)
    manifest = {
        "dataset_doi": DATASET_DOI,
        "dataset_url": DATASET_URL,
        "data_license": DATASET_LICENSE,
        "attribution": "Created by Mitsubishi Electric Research Laboratories (MERL), 2024; Giacomuzzo, Carli, Romeres, Dalla Libera",
        "archive_md5": DATASET_ARCHIVE_MD5,
        "split": split,
        "records": records,
        "reference_sha256": file_digest(output_path),
        "preparation_seconds": time.perf_counter() - start,
        "units": {"time": "s", "q": "rad", "qd": "rad/s", "qdd": "rad/s^2", "tau": "N*m"},
        "source_fields": {
            "time": "t",
            "q": "q_1..q_7",
            "qd": "dq_1..dq_7",
            "qdd": "ddq_1..ddq_7",
            "tau": "tau_interp_1..tau_interp_7",
        },
        "measurement_note": "Recorded ROS joint observations, publisher-filtered at 4 Hz. Acceleration is acausal velocity differentiation. tau_interp is the publisher's interpolated torque field used by its real-data estimator, not a recovered motor command. Manufacturer M/c/g and learned models are excluded.",
        "selection": "Three lowest numbered published training seeds, selected before fitting; all 16 published test seeds. No outcome-based sequence filtering.",
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_geometry(source_xml: Path, output_directory: Path) -> Path:
    """Copy geometry/kinematics while removing every authored dynamic parameter."""
    tree = ET.parse(source_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    mesh_directory = source_xml.parent / (compiler.get("meshdir", ".") if compiler is not None else ".")
    output_directory.mkdir(parents=True, exist_ok=True)
    assets = output_directory / "assets"
    assets.mkdir(exist_ok=True)
    copied = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag in (
                "inertial",
                "actuator",
                "keyframe",
                "contact",
                "equality",
                "sensor",
                "general",
                "motor",
                "position",
                "velocity",
                "intvelocity",
                "damper",
                "cylinder",
                "muscle",
                "adhesion",
            ):
                parent.remove(child)
        if parent.tag == "joint":
            for key in list(parent.attrib):
                if key not in {"name", "type", "axis", "pos", "range", "limited", "class", "group"}:
                    del parent.attrib[key]
        if parent.tag == "geom":
            for key in ("mass", "friction", "solref", "solimp", "margin", "gap"):
                parent.attrib.pop(key, None)
            parent.set("density", "0")
            parent.set("contype", "0")
            parent.set("conaffinity", "0")
        if parent.tag == "mesh" and "file" in parent.attrib:
            original = mesh_directory / parent.get("file")
            destination = assets / original.name
            shutil.copyfile(original, destination)
            parent.set("file", f"assets/{destination.name}")
            copied.append({"file": f"assets/{destination.name}", "sha256": file_digest(destination)})
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)
    output = output_directory / "panda_geometry.xml"
    tree.write(output, encoding="unicode")
    license_path = source_xml.parent / "LICENSE"
    if license_path.exists():
        shutil.copyfile(license_path, output_directory / "ASSET_LICENSE")
    (output_directory / "geometry-manifest.json").write_text(
        json.dumps(
            {
                "source": "https://github.com/google-deepmind/mujoco_menagerie",
                "revision": "8161bba264d7fa7c99ca301e91e7fb44737676ad",
                "source_xml_sha256": file_digest(source_xml),
                "geometry_xml_sha256": file_digest(output),
                "meshes": copied,
                "changes": "Remove authored inertial properties, actuator gains, damping, friction, armature, contacts, equality constraints, and initial configurations. Retain geometry, joint topology, transforms, and joint ranges. Collision disabled; all geometry has zero density. Runtime parameters are explicit homogeneous placeholders or agent-submitted values.",
            },
            indent=2,
        )
        + "\n"
    )
    return output


def main() -> None:
    """Prepare references separately from the sanitized geometry asset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--split", choices=("training", "heldout"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-source", type=Path)
    args = parser.parse_args()
    if args.geometry_source is not None:
        print(prepare_geometry(args.geometry_source, args.output))
    elif args.archive is not None and args.split is not None:
        print(json.dumps(prepare_reference(args.archive, args.output, split=args.split)))
    else:
        parser.error("Supply --geometry-source, or both --archive and --split")


if __name__ == "__main__":
    main()
