#!/usr/bin/env python3
"""
把单物种训练 CSV 转成 CodonTransformer 训练用 JSONL。

示例：
python scripts/prepare_one_species_json.py \
  --input_csv data/finetune_csv/chlamydomonas_reinhardtii.csv \
  --output_json data/finetune_json/chlamydomonas_reinhardtii_training_data.json
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare one species JSONL training file")
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"input csv not found: {in_path}")

    from CodonTransformer.CodonData import prepare_training_data

    prepare_training_data(str(in_path), str(out_path))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
