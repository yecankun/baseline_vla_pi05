#!/usr/bin/env bash
# USER-RUN long job. New paired training only; never resumes or overwrites.
set -euo pipefail
cd /home/zsw/project_2026
python=/home/zsw/miniconda3/envs/project2026-pi/bin/python
root=/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_bc_act_v1
preflight=simulation_output/pusht_bc_act_training_preflight_v1_retry2
for model in bc act; do
  if [ -e "$root/$model" ]; then
    echo "Existing $model run: inspect status/checkpoints and resume explicitly; refusing paired restart." >&2
    exit 1
  fi
  if [ ! -f "$preflight/$model/report.json" ]; then
    echo "Missing $model preflight report." >&2
    exit 1
  fi
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for model in bc act; do
  "$python" -B -u tools/train_pusht_bc_act.py --stage train --model "$model" --execute
done
