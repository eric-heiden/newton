# Live simulation evaluation

These tools compare an independent GPT-6 Astra agent using live Newton MCP
with an independent agent editing a Python configuration and starting a new
Newton process for every candidate. Both conditions use the same dynamics,
targets, physical limits, quality measurements and optional sensor images.
Nothing here assumes the live agent will win. Keep failures and report raw
usage and final independently verified quality.

Obtain the model sources from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
and [ManoSim](https://github.com/KevinyWu/manosim), using the exact revisions below:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git /path/to/mujoco_menagerie
git -C /path/to/mujoco_menagerie checkout 8161bba264d7fa7c99ca301e91e7fb44737676ad
git clone https://github.com/KevinyWu/manosim.git /path/to/manosim
git -C /path/to/manosim checkout 75c77c065c6979a6b251362fc2fe344d77737291
```

Use local assets obtained under their source licenses. No recordings, MANO
assets, or scanned meshes are redistributed here. Configure paths with:

```bash
export NEWTON_EVAL_MENAGERIE=/path/to/mujoco_menagerie
export NEWTON_EVAL_HUG_DATA=/path/to/data
export NEWTON_EVAL_MANOSIM=/path/to/manosim
```

Menagerie revision `8161bba264d7fa7c99ca301e91e7fb44737676ad` supplies
`franka_emika_panda/panda_nohand.xml` and `wonik_allegro/right_hand.xml`.
The HUG data root must contain `scenes/medium_2/aria_data.pkl` and
`hug_bench/test/medium_2/softball/sim_assets`. ManoSim revision
`75c77c065c6979a6b251362fc2fe344d77737291` supplies
`assets/myhand/mano_rhand/capsule_hand.xml` and its referenced meshes.
Record provenance manifests and SHA-256 hashes with every experiment.

Follow the [HUG project dataset and benchmark links](https://grasping.io/)
to obtain the source recording and scanned object assets. The related
[aria2mesh](https://github.com/KevinyWu/aria2mesh) and
[aria2mano](https://github.com/KevinyWu/aria2mano) projects document the object
and fitted-hand data pipelines. This harness consumes operator-provided
`aria_data.pkl` with `fps`, per-frame `index`, `T_world_device`, and right-hand
`T_device_wrist`, `pose`, and `landmarks` fields; it does not rerun fitting or
download recordings. Preserve the layout above when extracting the HUG scene
and simulation assets. Missing local assets skip the corresponding tests.

Prepare a fresh trial without launching an agent:

```bash
uv run --no-sync -m tools.mcp_evaluation.run_agents --scenario panda --condition restart --variant 0 --workspace /path/to/fresh-trial
```

Add `--run` to explicitly launch `codex exec --ignore-user-config --model
gpt-6-astra -c 'model_reasoning_effort="xhigh"' --json --ephemeral`.
Use a different workspace for each condition, task and variant. Trials have
a 600-second default budget and a 12-candidate instruction budget. Use the
same limits across paired conditions. Shell access is the same; baseline
candidates must each execute in a fresh simulation process. Agents may
batch candidates. Live candidates can batch validated parameter changes,
reset, step and metrics into one actual MCP `execute` request. Submission is
the final `config.py` in both conditions; a fresh process verifies it.

Use `--phase confirmation` with fresh paired workspaces after freezing the
implementation and prompts. The live harness selects the optional four-tool
`--profile code` interface: describe, execute, observe, and rebuild. Discovery
is optional; structured operations remain callable through `session.dispatch`
inside execute. Call observe directly for MCP image content. After an execution
failure, rebuild restores the same scenario and last validated configuration
with fresh state and measurements. A rebuild may also receive a `config` dict.
The public adapter's default profile still exposes all structured tools.

Final verification writes `verification/metrics.json` and its companion NPZ,
preserving the candidate outputs. Agent and server process groups are stopped
before verification on POSIX; Windows has parent-process cleanup only and is
not validated here. Summary candidate counts mean completed logged rollouts;
raw events retain partial and failed attempts. Trial source hashes cover the
harness, MCP implementation, and modified MuJoCo solver. MCP startup is limited
to 30 seconds and calls to 300 seconds; a running mutation that times out must
not be automatically retried because its outcome may be unknown.

Candidate outputs may use nested directories. Counts aggregate candidate and
process logs throughout the trial workspace, excluding `verification/`; summaries
include each relative log path and its record count for auditing.

The three tasks use CPU `SolverMuJoCo` with its native contacts. The
collision-pipeline contact tool is a separately generated diagnostic.
Sensor observations also run on CPU. On-disk compilation caches are shared
fairly; perform warmup before timed comparisons. Report live startup both
separately and added to agent time. The raw CLI JSONL and usage fields are
retained; cached input and reasoning output are subsets, not extra tokens.

Panda and Allegro retain imported geometry and inertia while replacing
authored actuators with Newton position drives. They assess tracking and
stability, not grasp success. Panda retains 87/12 N m joint effort limits;
Allegro uses a fixed 0.7 N m limit and a mounted base 0.25 m above the support
plane. Both follow a two-second smooth posture transition and one-second
hold. Task files publish thresholds before agents start.

HUG reconstructs a real contiguous three-second fitted MANO motion clip and
a scanned object. It composes `T_world_device @ T_device_wrist`, converts
rotations to xyzw, and applies the same translation to hand and object.
The support-plane height is inferred from the lowest placed mesh vertex.
The task starts with a wrong recording sample-rate setting and poor drive
parameters. Scoring compares against the original recording timeline;
changing playback speed cannot conceal tracking error. Wrist translation,
finger rotation, velocity stability and timing all have fixed thresholds.
There is no measured object trajectory or force ground truth, so this is a
physical replay/setup task, not a claim of reconstructing grasp success or
rerunning raw visual/MANO reconstruction. Fixed supplied hand morphology,
infinite support approximation, excluded hand/self and hand/plane contacts,
and an explicit free-wrist drive are disclosed in provenance.

The current importer cannot combine this MJCF's three slides and ball wrist
without an indexed custom-attribute error. The ingest recipe normalizes them
to an equivalent free joint while retaining component damping and armature;
the resulting force drive has documented caps. Source assets remain untouched.

Run correctness checks with local assets installed:

```bash
uv run --no-sync -m unittest tools.mcp_evaluation.test_scenarios
```

Render actual saved rollout states without rerunning dynamics:

```bash
uv run --no-sync -m tools.mcp_evaluation.render_trace /path/to/metrics.json --output /path/to/frames
```

The output PNG sequence includes a playback-time manifest; encoding video is
optional. A separate scripted overhead experiment uses actual local requests
and matched fresh processes; it measures systems cost, not agent efficiency:

```bash
uv run --no-sync -m tools.mcp_evaluation.microbenchmark --scenario panda --output /path/to/fresh-benchmark
```
