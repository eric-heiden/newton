# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Convert DFKI double-pendulum system-identification records into task CSVs.

Source: https://github.com/dfki-ric-underactuated-lab/double_pendulum (BSD-3-Clause),
``data/experiment_records/design_C.0/20220914/sys_id/trajectory_XX.csv``.
Runs 00-09 become ``train_XX.csv`` for the agent; runs 10 and 11 are held out.

Usage: ``python prepare_data.py DFKI_DATA_DIR TRAIN_DIR HELDOUT_DIR``
"""

import sys
from pathlib import Path

import numpy as np

HEADER = "t,q1,q2,qd1,qd2,tau1,tau2"


def convert(source: Path) -> np.ndarray:
    data = np.genfromtxt(source, delimiter=",", names=True)
    columns = ("time", "pos_meas1", "pos_meas2", "vel_meas1", "vel_meas2", "tau_meas1", "tau_meas2")
    return np.stack([data[name] for name in columns], axis=1)


def main() -> None:
    source, train, heldout = (Path(arg) for arg in sys.argv[1:4])
    records = source / "experiment_records/design_C.0/20220914/sys_id"
    for directory in (train, heldout):
        directory.mkdir(parents=True, exist_ok=True)
    for index in range(12):
        target = (train / f"train_{index:02d}.csv") if index < 10 else (heldout / f"heldout_{index:02d}.csv")
        rows = convert(records / f"trajectory_{index:02d}.csv")
        np.savetxt(target, rows, delimiter=",", header=HEADER, comments="", fmt="%.6g")


if __name__ == "__main__":
    main()
