"""
time_icr_benchmark.py
重新跑ICR对52条benchmark的推理，记录总耗时。
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from mocps.config import BASE_MODEL_DIR, CHECKPOINTS_DIR, DEFAULT_HUMAN_EVAL_CSV, PROJECT_ROOT
from mocps.metrics import compute_mfe
from mocps.sequences import normalize_dna, normalize_protein

ORGANISM = "Homo sapiens"


def parse_args():
    parser = argparse.ArgumentParser(description="Time ICR inference on the 52-protein benchmark")
    parser.add_argument(
        "--xlsx",
        default=str(PROJECT_ROOT / "data" / "Fig. 4, Supplementary Fig. 19f-h, 20, 21, 22 and 25.xlsx"),
        help="Benchmark Excel file path",
    )
    parser.add_argument("--sheet", default="DNA sequences, CSI, CIS-element", help="Excel sheet name")
    parser.add_argument("--ref_csv", default=str(DEFAULT_HUMAN_EVAL_CSV), help="Reference evaluation CSV")
    parser.add_argument("--base_model", default=str(BASE_MODEL_DIR), help="Base CodonTransformer model directory")
    parser.add_argument("--checkpoint", default=str(CHECKPOINTS_DIR / "homo_sapiens" / "finetune.ckpt"))
    parser.add_argument("--organism", default=ORGANISM, help="Target organism")
    parser.add_argument("--n_rounds", type=int, default=5)
    parser.add_argument("--top_k_frac", type=float, default=0.15)
    parser.add_argument("--mfe_candidates", type=int, default=5, help="Total candidates including the ICR sequence")
    return parser.parse_args()


def main():
    args = parse_args()

    from CodonTransformer.CodonData import get_codon_frequencies
    from CodonTransformer.CodonEvaluation import get_CSI_weights
    from CodonTransformer.CodonPrediction import predict_dna_sequence
    from mocps.icr import ICROptimizer
    from mocps.modeling import load_tokenizer_and_model

    print("加载数据和模型...")
    benchmark = pd.read_excel(args.xlsx, sheet_name=args.sheet)
    human_bm = benchmark[benchmark["organism"] == args.organism].copy().reset_index(drop=True)

    ref_df = pd.read_csv(args.ref_csv)
    ref_dnas = ref_df["natural_dna"].head(500).tolist()
    codon_freq = get_codon_frequencies(dna_sequences=ref_dnas, organism=args.organism)
    csi_weights = get_CSI_weights(ref_dnas)

    tokenizer, model, device = load_tokenizer_and_model(
        args.base_model,
        checkpoint=args.checkpoint,
        attention_type="original_full",
    )

    optimizer = ICROptimizer(
        model=model,
        tokenizer=tokenizer,
        organism=args.organism,
        codon_frequencies=codon_freq,
        csi_weights=csi_weights,
        device=device,
        w_csi=2.0,
        w_cfd=1.0,
        w_cis=2.0,
    )

    print(f"开始计时 ICR 推理（{len(human_bm)}条）...")
    t_start = time.time()
    success = 0

    for idx, row in human_bm.iterrows():
        protein = normalize_protein(str(row["protein sequence"]))
        init_dna = normalize_dna(str(row["finetune_transformer_dna"]))
        print(f"  [{idx + 1}/{len(human_bm)}] {row['protein']}", end="", flush=True)

        try:
            icr_dna, _ = optimizer.optimize(
                protein=protein,
                initial_dna=init_dna,
                n_rounds=args.n_rounds,
                top_k_frac=args.top_k_frac,
            )
            candidates = [icr_dna]
            for _ in range(max(args.mfe_candidates - 1, 0)):
                try:
                    result = predict_dna_sequence(
                        protein=protein,
                        organism=args.organism,
                        device=device,
                        tokenizer=tokenizer,
                        model=model,
                        attention_type="original_full",
                        deterministic=False,
                        temperature=0.8,
                        top_p=0.95,
                        match_protein=True,
                    )
                    candidate = normalize_dna(result.predicted_dna)
                    candidate, _ = optimizer.optimize(
                        protein=protein,
                        initial_dna=candidate,
                        n_rounds=3,
                        top_k_frac=0.10,
                    )
                    candidates.append(candidate)
                except Exception:
                    pass

            mfe_scores = [compute_mfe(candidate) for candidate in candidates]
            _ = candidates[mfe_scores.index(min(mfe_scores))]
            success += 1
            elapsed_so_far = time.time() - t_start
            print(f"  累计耗时={elapsed_so_far:.1f}s")
        except Exception as exc:
            print(f"  {exc}")

    t_total = time.time() - t_start
    t_per = t_total / success if success > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"ICR推理计时结果（{success}条序列）")
    print(f"{'=' * 50}")
    print(f"  总耗时:     {t_total:.1f} 秒")
    print(f"  每条耗时:   {t_per:.1f} 秒/条")
    print("\n** Use the total runtime in the ICR inference-time result table **")
    print("\n对比参考:")
    print("  CT Fine-tuned:     89.7 秒")
    print("  Expert Iteration:  98.5 秒")
    print(f"  ICR (Ours):        {t_total:.1f} 秒  ← 本次结果")


if __name__ == "__main__":
    main()
