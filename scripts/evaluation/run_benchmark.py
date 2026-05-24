import argparse
from pathlib import Path

import pandas as pd

from mocps.config import (
    BASE_MODEL_DIR,
    DEFAULT_HUMAN_CHECKPOINT,
    DEFAULT_HUMAN_EVAL_CSV,
    DEFAULT_ORGANISM,
    PROJECT_ROOT,
    RESULTS_DIR,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run commercial benchmark with ICR optimization")
    parser.add_argument(
        "--xlsx",
        default=str(PROJECT_ROOT / "data" / "Fig. 4, Supplementary Fig. 19f-h, 20, 21, 22 and 25.xlsx"),
        help="Benchmark Excel file path",
    )
    parser.add_argument("--sheet", default="DNA sequences, CSI, CIS-element", help="Excel sheet name")
    parser.add_argument("--organism", default=DEFAULT_ORGANISM, help="Target organism")
    parser.add_argument("--ref_csv", default=str(DEFAULT_HUMAN_EVAL_CSV), help="Reference evaluation CSV")
    parser.add_argument("--base_model", default=str(BASE_MODEL_DIR), help="Base CodonTransformer model directory")
    parser.add_argument("--checkpoint", default=str(DEFAULT_HUMAN_CHECKPOINT), help="Fine-tuned checkpoint path")
    parser.add_argument("--output_csv", default=str(RESULTS_DIR / "homo_sapiens_benchmark_icr.csv"))
    parser.add_argument("--icr_rounds", type=int, default=5)
    parser.add_argument("--top_k_frac", type=float, default=0.15)
    parser.add_argument("--mfe_candidates", type=int, default=5, help="Total candidates including the ICR sequence")
    return parser.parse_args()


def main():
    args = parse_args()

    from CodonTransformer.CodonData import get_codon_frequencies
    from CodonTransformer.CodonEvaluation import get_CSI_weights
    from CodonTransformer.CodonPrediction import predict_dna_sequence
    from mocps.icr import ICROptimizer
    from mocps.metrics import compute_cfd, compute_cis, compute_csi, compute_mfe
    from mocps.modeling import load_tokenizer_and_model
    from mocps.sequences import normalize_dna, normalize_protein

    benchmark = pd.read_excel(args.xlsx, sheet_name=args.sheet)
    human_bm = benchmark[benchmark["organism"] == args.organism].copy().reset_index(drop=True)
    print(f"{args.organism} benchmark: {len(human_bm)} 条")

    ref_df = pd.read_csv(args.ref_csv)
    ref_dnas = ref_df["natural_dna"].head(500).tolist()
    codon_freq = get_codon_frequencies(dna_sequences=ref_dnas, organism=args.organism)
    csi_weights = get_CSI_weights(ref_dnas)

    tokenizer, model, device = load_tokenizer_and_model(
        base_model=args.base_model,
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

    results = []
    for idx, row in human_bm.iterrows():
        protein = normalize_protein(str(row["protein sequence"]))
        init_dna = normalize_dna(str(row["finetune_transformer_dna"]))
        print(f"\n[{idx + 1}/{len(human_bm)}] {row['protein']} (长度={len(protein)})")

        if len(protein) > 2040 or len(init_dna) < 3:
            print("  跳过（序列过长）")
            continue

        icr_dna, _ = optimizer.optimize(
            protein=protein,
            initial_dna=init_dna,
            n_rounds=args.icr_rounds,
            top_k_frac=args.top_k_frac,
        )

        candidates = [icr_dna]
        for _ in range(max(0, args.mfe_candidates - 1)):
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
                cand = normalize_dna(result.predicted_dna)
                cand, _ = optimizer.optimize(
                    protein=protein,
                    initial_dna=cand,
                    n_rounds=3,
                    top_k_frac=0.10,
                )
                candidates.append(cand)
            except Exception:
                pass

        mfe_scores = [compute_mfe(c) for c in candidates]
        best_idx = mfe_scores.index(min(mfe_scores))
        final_dna = candidates[best_idx]
        final_mfe = mfe_scores[best_idx]
        print(f"  MFE筛选: {len(candidates)}条候选, 最优={final_mfe:.2f}")

        results.append({
            "protein": row["protein"],
            "organism": args.organism,
            "finetune_CSI": row["finetune_transformer_CSI"],
            "finetune_CIS": row["finetune_transformer_CIS"],
            "pretrain_CSI": row["pretrain_transformer_CSI"],
            "pretrain_CIS": row["pretrain_transformer_CIS"],
            "twist_CSI": row["twist_CSI"],
            "twist_CIS": row["twist_CIS"],
            "idt_CSI": row["idt_CSI"],
            "idt_CIS": row["idt_CIS"],
            "genewiz_CSI": row["genewiz_CSI"],
            "genewiz_CIS": row["genewiz_CIS"],
            "icr_dna": final_dna,
            "icr_CSI": compute_csi(final_dna, csi_weights),
            "icr_CFD": compute_cfd(final_dna, codon_freq) / 100,
            "icr_CIS": compute_cis(final_dna),
            "icr_MFE": final_mfe,
            "init_MFE": compute_mfe(init_dna),
            "init_CSI": compute_csi(init_dna, csi_weights),
            "init_CIS": compute_cis(init_dna),
        })

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)
    print(f"\n完成！保存至 {output_csv}")

    print("\n=== 汇总对比 ===")
    print(f"{'方法':<20} {'CSI':>8} {'CIS':>8}")
    print("-" * 40)
    for method, csi_col, cis_col in [
        ("Twist", "twist_CSI", "twist_CIS"),
        ("IDT", "idt_CSI", "idt_CIS"),
        ("Genewiz", "genewiz_CSI", "genewiz_CIS"),
        ("CT fine-tuned", "finetune_CSI", "finetune_CIS"),
        ("ICR (Ours)", "icr_CSI", "icr_CIS"),
    ]:
        print(f"{method:<20} {out[csi_col].mean():>8.4f} {out[cis_col].mean():>8.2f}")


if __name__ == "__main__":
    main()
