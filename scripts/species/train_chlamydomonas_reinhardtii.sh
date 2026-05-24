#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/root/autodl-tmp/CodonTransformer_base}"
SPECIES="chlamydomonas_reinhardtii"
CHECKPOINT_DIR="${PROJECT_ROOT}/artifacts/checkpoints/${SPECIES}"
JSON_DIR="${PROJECT_ROOT}/data/finetune_json"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate codon

mkdir -p "${JSON_DIR}" "${CHECKPOINT_DIR}"

python "${SCRIPT_DIR}/prepare_one_species_json.py" \
  --input_csv "${PROJECT_ROOT}/data/finetune_csv/${SPECIES}.csv" \
  --output_json "${JSON_DIR}/${SPECIES}_training_data.json"

python "${PROJECT_ROOT}/scripts/training/finetune.py" \
  --dataset_json "${JSON_DIR}/${SPECIES}_training_data.json" \
  --base_model_dir "${BASE_MODEL_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --checkpoint_filename finetune.ckpt \
  --batch_size 4 \
  --max_epochs 15 \
  --num_workers 0 \
  --accumulate_grad_batches 2 \
  --num_gpus 1 \
  --learning_rate 0.00005 \
  --warmup_fraction 0.1 \
  --save_every_n_steps 512 \
  --seed 23 2>&1 | tee "${CHECKPOINT_DIR}/train.log"

echo "Checkpoint:"
ls -lh "${CHECKPOINT_DIR}/finetune.ckpt"

echo "Metrics CSV:"
find "${CHECKPOINT_DIR}/lightning_logs" -name metrics.csv -print
