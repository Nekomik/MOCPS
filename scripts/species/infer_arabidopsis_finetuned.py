#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd

from mocps.config import BASE_MODEL_DIR, CHECKPOINTS_DIR, DATA_DIR, RESULTS_DIR
from mocps.sequences import dna_to_protein, normalize_dna, normalize_protein

DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "arabidopsis_thaliana" / "finetune.ckpt"
DEFAULT_INPUT_CSV = DATA_DIR / "finetune_eval_csv" / "arabidopsis_thaliana_eval.csv"
DEFAULT_OUTPUT_CSV = RESULTS_DIR / "arabidopsis_thaliana_with_my_finetune.csv"

TARGET_ORG = "Arabidopsis thaliana"
ATTENTION_TYPE = "original_full"
DETERMINISTIC = True


def parse_args():
    parser = argparse.ArgumentParser(description="Run Arabidopsis fine-tuned inference on eval CSV")
    parser.add_argument("--base_model", default=str(BASE_MODEL_DIR), help="Local base model directory")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Fine-tuned state_dict checkpoint")
    parser.add_argument("--input_csv", default=str(DEFAULT_INPUT_CSV), help="Input evaluation CSV")
    parser.add_argument("--output_csv", default=str(DEFAULT_OUTPUT_CSV), help="Output CSV with generated DNA")
    parser.add_argument("--organism", default=TARGET_ORG, help="Target organism name")
    parser.add_argument("--attention_type", default=ATTENTION_TYPE, help="BigBird attention type")
    parser.add_argument("--non_deterministic", action="store_true", help="Use stochastic prediction")
    return parser.parse_args()


def main():
    args = parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)
    checkpoint_path = Path(args.checkpoint)
    base_model_path = Path(args.base_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"input csv not found: {in_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not base_model_path.exists():
        raise FileNotFoundError(f"base model dir not found: {base_model_path}")

    from CodonTransformer.CodonPrediction import predict_dna_sequence
    from mocps.modeling import load_tokenizer_and_model

    tokenizer, model, device = load_tokenizer_and_model(
        base_model_path,
        checkpoint=checkpoint_path,
        attention_type=args.attention_type,
    )

    df = pd.read_csv(in_path)

    if "natural_dna" not in df.columns and "dna" in df.columns:
        df = df.rename(columns={"dna": "natural_dna"})

    required = {"protein", "organism", "natural_dna"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")

    df = df.copy()
    df["protein"] = df["protein"].map(normalize_protein)
    df["natural_dna"] = df["natural_dna"].map(normalize_dna)

    df = df.loc[df["organism"] == args.organism].copy()
    if df.empty:
        raise ValueError(f"no rows for organism: {args.organism}")

    my_dna = []
    valid_triplet = []
    has_stop = []
    translated_match = []

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        protein = row["protein"]

        pred = predict_dna_sequence(
            protein=protein,
            organism=args.organism,
            device=device,
            tokenizer=tokenizer,
            model=model,
            attention_type=args.attention_type,
            deterministic=not args.non_deterministic,
        )

        dna = normalize_dna(pred.predicted_dna)
        my_dna.append(dna)

        is_triplet = len(dna) % 3 == 0
        stop_ok = dna[-3:] in {"TAA", "TAG", "TGA"} if len(dna) >= 3 else False
        prot_ok = dna_to_protein(dna) == protein if is_triplet else False

        valid_triplet.append(is_triplet)
        has_stop.append(stop_ok)
        translated_match.append(prot_ok)

        if i % 50 == 0 or i == total:
            print(f"[{i}/{total}] done")

    df["my_finetune_dna"] = my_dna
    df["my_valid_triplet"] = valid_triplet
    df["my_has_stop"] = has_stop
    df["my_translated_match"] = translated_match

    df.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
