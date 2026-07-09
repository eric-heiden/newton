#!/usr/bin/env bash
# Full HRDexDB experiment pipeline (resumable stages).
set -uo pipefail
cd "$(dirname "$0")"

run() { echo "=== $* ==="; uv run python "$@" 2>&1 | grep -vE "Module |overflow|exceeded|UserWarning|warnings.warn|CoACD.*info"; }

# 1. Pilot: PD target source study
for hand in allegro_v5 inspire_f1; do
  for src in cmd meas; do
    run evaluate.py --hand "$hand" --objects apple banana baseball book blue_plastic_box beige_brush --tag pilot --target-source "$src"
  done
done

# 2. Broad default evaluation
for hand in allegro_v5 inspire_f1; do
  run evaluate.py --hand "$hand" --tag default --target-source cmd
done

# 3. CMA-ES tuning per hand
for hand in allegro_v5 inspire_f1; do
  if [ ! -f "results/tuned_params_${hand}_cmd.json" ]; then
    run tune.py --hand "$hand" --target-source cmd --popsize 8 --maxiter 22 --budget-hours 3
  fi
done

# 4. Broad tuned evaluation
for hand in allegro_v5 inspire_f1; do
  run evaluate.py --hand "$hand" --tag tuned --target-source cmd --params "results/tuned_params_${hand}_cmd.json"
done

# 5. Revalidation + repeatability
run revalidate.py
run repeatability.py --runs 8

# 6. Figures
run make_plots.py

echo "=== pipeline done ==="
