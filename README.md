# MOCPS

MOCPS (Multi-objective Codon Optimization with Policy-guided Search) is a research codebase for multi-objective codon optimization. It extends CodonTransformer with inference-time synonymous codon search, RNA folding based post-selection, and expert-iteration training utilities.

The repository is intended to contain source code, lightweight examples, and reproducible scripts only. Large model checkpoints, full datasets, generated result tables, and local documents are excluded from version control.

## Method Overview

MOCPS separates codon optimization into interpretable stages:

- **CT generation**: use a fine-tuned CodonTransformer checkpoint to generate an initial host-adapted coding DNA sequence.
- **ICR search**: refine the sequence through synonymous codon replacement guided by multi-objective rewards.
- **MFE post-selection**: optionally select candidates with lower predicted RNA folding free energy using ViennaRNA.
- **Expert iteration**: distill search-improved sequences into a model checkpoint for direct generation.

The current implementation focuses on Homo sapiens and uses CSI, CFD, CIS, MFE, and nMFE as computational evaluation signals.

## Repository Layout

```text
CodonTransformer/          # Upstream CodonTransformer package code used by this project
mocps/                     # MOCPS metrics, sequence utilities, ICR search, model loading
scripts/                   # Training, evaluation, species-specific, and helper scripts
slurm/                     # Example cluster job scripts
data/                      # Small example data only; large datasets are ignored
requirements.txt           # Python dependencies
setup.py                   # Package setup entry inherited from CodonTransformer
```

Script categories are documented in `scripts/README.md`.

## Files Not Tracked

The following local artifacts are intentionally excluded by `.gitignore`:

```text
artifacts/                 # checkpoints, logs, full evaluation outputs
CodonTransformer_base/     # base model directory
data/finetune_*            # large training/evaluation datasets
release/                   # local archives
*.ckpt, *.safetensors      # model weights
```

These files should be distributed separately if they are needed to reproduce a specific experiment.

## Installation

A Conda environment is recommended:

```bash
conda create -n codon python=3.10 -y
conda activate codon
pip install -r requirements.txt
```

Install ViennaRNA separately if MFE calculation is required:

```bash
conda install -c bioconda viennarna -y
```

GPU is recommended for model inference, fine-tuning, and benchmark evaluation.

## Required Local Artifacts

Some scripts expect local model and evaluation files. Default paths are defined in `mocps/config.py`:

```text
CodonTransformer_base/
artifacts/checkpoints/homo_sapiens/finetune.ckpt
artifacts/checkpoints/homo_sapiens_expert_iter/expert_iter.ckpt
artifacts/results/all_eval_with_my_finetune/homo_sapiens_all_eval_with_my_finetune.csv
```

The paths can be overridden with environment variables:

```bash
export MOCPS_BASE_MODEL_DIR=/path/to/CodonTransformer_base
export MOCPS_HUMAN_CHECKPOINT=/path/to/finetune.ckpt
export MOCPS_EXPERT_CHECKPOINT=/path/to/expert_iter.ckpt
export MOCPS_HUMAN_EVAL_CSV=/path/to/homo_sapiens_all_eval_with_my_finetune.csv
```

## Command-line Inference

Example:

```bash
python scripts/utils/predict.py \
  --protein "FVNQHLCGSHLVEALYLVCGERGFFYTPKT" \
  --n_rounds 2 \
  --use_mfe
```

The script loads the configured CodonTransformer checkpoint, generates an initial DNA sequence, applies ICR refinement, and optionally performs MFE candidate selection.

## Training and Evaluation Scripts

Common entry points:

```text
scripts/training/finetune.py                         # fine-tune CodonTransformer
scripts/training/expert_iter_finetune.py             # expert-iteration fine-tuning
scripts/utils/batch_prepare_finetune_9species.py      # prepare multi-species fine-tuning data
scripts/evaluation/batch_infer_eval_9species.py       # batch inference and evaluation
scripts/evaluation/run_benchmark.py                  # benchmark evaluation
scripts/evaluation/run_benchmark_expert.py           # expert checkpoint evaluation
```

Most scripts assume local `data/`, `artifacts/`, and checkpoint paths. Adjust paths, batch sizes, and GPU settings before running large jobs.

## Development Checks

```bash
python -m py_compile $(find mocps scripts -name '*.py')
git status --short
```

Before publishing, verify that large files are ignored:

```bash
git check-ignore -v artifacts/checkpoints/homo_sapiens/finetune.ckpt
git check-ignore -v CodonTransformer_base/model.safetensors
git check-ignore -v data/finetune_all_eval_csv
```

## Relationship to CodonTransformer

This project extends CodonTransformer with additional optimization and evaluation code. The upstream CodonTransformer source and model are developed by the original authors and remain subject to their license and citation requirements.

## License

This repository keeps the upstream Apache-2.0 license. Checkpoint files and datasets may have separate terms depending on their source.
