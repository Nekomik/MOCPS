
"""
batch_infer_eval_9species.py

用途：
1. 扫描 data/finetune_all_eval_csv/*_all_eval.csv
2. 加载对应 artifacts/checkpoints/<species>/finetune.ckpt
3. 在原表所有列基础上新增：
   - my_finetune_dna
   - my_valid_triplet
   - my_has_stop
   - my_translated_match
4. 输出到 artifacts/results/all_eval_with_my_finetune/<species>_all_eval_with_my_finetune.csv

说明：
- 默认推理全部物种，可通过 --species 指定一个或多个物种 stem
- 对 chloroplast 增加 checkpoint 别名兼容
"""

import argparse
from pathlib import Path

import pandas as pd

from mocps.config import BASE_MODEL_DIR, CHECKPOINTS_DIR, DATA_DIR, RESULTS_DIR

ALL_EVAL_DIR = DATA_DIR / "finetune_all_eval_csv"
OUTPUT_DIR = RESULTS_DIR / "all_eval_with_my_finetune"


def checkpoint_candidates(stem: str):
    cands = [stem]

    if stem == "chlamydomonas_reinhardtii_chloroplast":
        cands += [
            "chloroplast_of_c_reinhardtii",
            "chloroplast_of_chlamydomonas_reinhardtii",
        ]

    if stem == "chloroplast_of_c_reinhardtii":
        cands += [
            "chlamydomonas_reinhardtii_chloroplast",
            "chloroplast_of_chlamydomonas_reinhardtii",
        ]

    if stem == "escherichia_coli_general":
        cands += ["escherichia_coli"]

    # 去重并保持顺序
    out = []
    for x in cands:
        if x not in out:
            out.append(x)
    return out


def resolve_checkpoint(stem: str):
    for cand in checkpoint_candidates(stem):
        ckpt = CHECKPOINTS_DIR / cand / "finetune.ckpt"
        if ckpt.exists():
            return ckpt, cand
    return None, None


def infer_one_species(eval_csv: Path, checkpoint_path: Path, output_csv: Path):
    from CodonTransformer.CodonPrediction import predict_dna_sequence
    from mocps.modeling import load_tokenizer_and_model
    from mocps.sequences import dna_to_protein, normalize_dna, normalize_protein

    df = pd.read_csv(eval_csv)

    if "natural_dna" not in df.columns and "dna" in df.columns:
        df = df.rename(columns={"dna": "natural_dna"})

    required = {"protein", "organism", "natural_dna"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{eval_csv.name}: missing columns: {missing}")

    df = df.copy()
    df["protein"] = df["protein"].map(normalize_protein)
    df["natural_dna"] = df["natural_dna"].map(normalize_dna)

    tokenizer, model, device = load_tokenizer_and_model(
        base_model=BASE_MODEL_DIR,
        checkpoint=checkpoint_path,
        attention_type="original_full",
    )

    my_dna = []
    valid_triplet = []
    has_stop = []
    translated_match = []

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        protein = row["protein"]
        organism = row["organism"]

        pred = predict_dna_sequence(
            protein=protein,
            organism=organism,
            device=device,
            tokenizer=tokenizer,
            model=model,
            attention_type="original_full",
            deterministic=True,
        )

        dna = normalize_dna(pred.predicted_dna)
        my_dna.append(dna)

        is_triplet = (len(dna) % 3 == 0)
        stop_ok = dna[-3:] in {"TAA", "TAG", "TGA"} if len(dna) >= 3 else False
        prot_ok = (dna_to_protein(dna) == protein) if is_triplet else False

        valid_triplet.append(is_triplet)
        has_stop.append(stop_ok)
        translated_match.append(prot_ok)

        if i % 50 == 0 or i == total:
            print(f"[{eval_csv.stem}] [{i}/{total}] done")

    df["my_finetune_dna"] = my_dna
    df["my_valid_triplet"] = valid_triplet
    df["my_has_stop"] = has_stop
    df["my_translated_match"] = translated_match

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"saved: {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for fine-tuned species checkpoints")
    parser.add_argument("--all_eval_dir", default=str(ALL_EVAL_DIR), help="Directory containing *_all_eval.csv files")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), help="Directory for augmented evaluation CSV files")
    parser.add_argument(
        "--species",
        nargs="*",
        default=None,
        help="Species stems to run, for example: homo_sapiens mus_musculus. Omit to run all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    all_eval_dir = Path(args.all_eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_files = sorted(all_eval_dir.glob("*_all_eval.csv"))
    if not eval_files:
        raise FileNotFoundError(f"no *_all_eval.csv found under: {all_eval_dir}")

    if args.species is not None:
        wanted = set(args.species)
        eval_files = [p for p in eval_files if p.name.replace("_all_eval.csv", "") in wanted]

    if not eval_files:
        raise FileNotFoundError("after filtering selected species, no eval files remain")

    summary = []

    for eval_csv in eval_files:
        stem = eval_csv.name.replace("_all_eval.csv", "")
        checkpoint_path, checkpoint_stem = resolve_checkpoint(stem)
        output_csv = output_dir / f"{stem}_all_eval_with_my_finetune.csv"

        if checkpoint_path is None:
            print(f"[SKIP] checkpoint missing for {stem}")
            print(f"        tried: {checkpoint_candidates(stem)}")
            continue

        print("" \
        "" + "=" * 80)
        print(f"[START] {stem}")
        print(f"[CHECKPOINT] {checkpoint_path}")
        if checkpoint_stem != stem:
            print(f"[NOTE] using checkpoint alias folder: {checkpoint_stem}")
        print("=" * 80)

        infer_one_species(
            eval_csv=eval_csv,
            checkpoint_path=checkpoint_path,
            output_csv=output_csv,
        )

        summary.append(
            {
                "species": stem,
                "input_eval_csv": str(eval_csv),
                "checkpoint": str(checkpoint_path),
                "output_csv": str(output_csv),
            }
        )

    summary_path = output_dir / "batch_infer_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
