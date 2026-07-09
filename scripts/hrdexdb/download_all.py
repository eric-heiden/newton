# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Download the HRDexDB episode subset used by the experiments.

For each hand and object, downloads up to ``--scenes-per-object`` complete
episodes (skipping episodes with missing raw data or calibration) plus the
object mesh. Writes a manifest json of what is locally available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import DATA_ROOT, HANDS, download_episode, download_mesh, list_remote_objects, list_remote_scenes

MANIFEST = Path(__file__).parent / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=int, default=35, help="max objects per hand")
    parser.add_argument("--scenes-per-object", type=int, default=2)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    common = sorted(set(list_remote_objects(HANDS[0])) & set(list_remote_objects(HANDS[1])))
    objects = common[: args.objects]
    print(f"{len(common)} common objects; using {len(objects)}")

    for hand in HANDS:
        manifest.setdefault(hand, {})
        for obj in objects:
            have = manifest[hand].get(obj, [])
            if len(have) >= args.scenes_per_object:
                continue
            try:
                scenes = list_remote_scenes(hand, obj)
            except Exception as e:
                print(f"{hand}/{obj}: listing failed ({e})")
                continue
            got = list(have)
            for scene in scenes:
                if len(got) >= args.scenes_per_object:
                    break
                if scene in got:
                    continue
                try:
                    if download_episode(hand, obj, scene):
                        got.append(scene)
                        print(f"{hand}/{obj}/{scene}: ok")
                    else:
                        print(f"{hand}/{obj}/{scene}: incomplete, skipped")
                except Exception as e:
                    print(f"{hand}/{obj}/{scene}: failed ({e})")
            if got:
                try:
                    download_mesh(obj)
                except Exception as e:
                    print(f"mesh {obj}: failed ({e})")
                    got = []
            manifest[hand][obj] = got
            MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True))

    n = {h: sum(len(v) for v in manifest.get(h, {}).values()) for h in HANDS}
    print(f"manifest: {n} episodes, root {DATA_ROOT}")


if __name__ == "__main__":
    main()
