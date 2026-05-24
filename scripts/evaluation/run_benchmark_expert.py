"""
run_benchmark_expert.py
在52条benchmark上评测 CT Fine-tuned、ICR 和 Expert Iteration。
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from mocps.config import BASE_MODEL_DIR, CHECKPOINTS_DIR, DEFAULT_HUMAN_EVAL_CSV, PROJECT_ROOT, RESULTS_DIR
from mocps.metrics import compute_cfd, compute_cis, compute_csi, compute_mfe
from mocps.sequences import normalize_dna, normalize_protein

ORGANISM = "Homo sapiens"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark fine-tuned and expert-iteration models")
    parser.add_argument(
        "--xlsx",
        default=str(PROJECT_ROOT / "data" / "Fig. 4, Supplementary Fig. 19f-h, 20, 21, 22 and 25.xlsx"),
        help="Benchmark Excel file path",
    )
    parser.add_argument("--sheet", default="DNA sequences, CSI, CIS-element", help="Excel sheet name")
    parser.add_argument("--ref_csv", default=str(DEFAULT_HUMAN_EVAL_CSV), help="Reference evaluation CSV")
    parser.add_argument("--base_model", default=str(BASE_MODEL_DIR), help="Base CodonTransformer model directory")
    parser.add_argument("--finetune_checkpoint", default=str(CHECKPOINTS_DIR / "homo_sapiens" / "finetune.ckpt"))
    parser.add_argument("--expert_checkpoint", default=str(CHECKPOINTS_DIR / "homo_sapiens_expert_iter" / "expert_iter.ckpt"))
    parser.add_argument("--icr_result_csv", default=str(RESULTS_DIR / "homo_sapiens_benchmark_icr.csv"))
    parser.add_argument("--output_csv", default=str(RESULTS_DIR / "homo_sapiens_benchmark_full.csv"))
    parser.add_argument("--output_ft_csv", default=str(RESULTS_DIR / "benchmark_ft.csv"))
    parser.add_argument("--output_expert_csv", default=str(RESULTS_DIR / "benchmark_expert_iter.csv"))
    parser.add_argument("--organism", default=ORGANISM, help="Target organism")
    return parser.parse_args()


def main():
    args = parse_args()

    from CodonTransformer.CodonData import get_codon_frequencies
    from CodonTransformer.CodonEvaluation import get_CSI_weights
    from CodonTransformer.CodonPrediction import predict_dna_sequence
    from mocps.modeling import load_tokenizer_and_model

    print("读取benchmark数据...")
    benchmark = pd.read_excel(args.xlsx, sheet_name=args.sheet)
    human_bm = benchmark[benchmark["organism"] == args.organism].copy().reset_index(drop=True)
    print(f"  共 {len(human_bm)} 条")

    ref_df = pd.read_csv(args.ref_csv)
    ref_dnas = ref_df["natural_dna"].head(500).tolist()
    codon_freq = get_codon_frequencies(dna_sequences=ref_dnas, organism=args.organism)
    csi_weights = get_CSI_weights(ref_dnas)

    tokenizer, _, device = load_tokenizer_and_model(args.base_model, eval_mode=True)

    def load_model(ckpt_path):
        _, loaded_model, _ = load_tokenizer_and_model(
            args.base_model,
            device=device,
            checkpoint=ckpt_path,
            attention_type="original_full",
        )
        return loaded_model

    def eval_metrics(dna):
        dna = normalize_dna(dna)
        return {
            "CSI": compute_csi(dna, csi_weights),
            "CFD": compute_cfd(dna, codon_freq),
            "CIS": compute_cis(dna),
            "MFE": compute_mfe(dna),
        }

    def infer_all(model, label):
        results = []
        t0 = time.time()
        for idx, row in human_bm.iterrows():
            protein = normalize_protein(str(row["protein sequence"]))
            print(f"  [{label}] [{idx + 1}/{len(human_bm)}] {row['protein']}", end="", flush=True)
            try:
                result = predict_dna_sequence(
                    protein=protein,
                    organism=args.organism,
                    device=device,
                    tokenizer=tokenizer,
                    model=model,
                    attention_type="original_full",
                    deterministic=True,
                    match_protein=True,
                )
                dna = normalize_dna(result.predicted_dna)
                metrics = eval_metrics(dna)
                print(f"  CSI={metrics['CSI']:.4f} CIS={metrics['CIS']} MFE={metrics['MFE']:.1f}")
                results.append({"protein": row["protein"], **metrics})
            except Exception as exc:
                print(f"  {exc}")
        elapsed = time.time() - t0
        return pd.DataFrame(results), elapsed

    print(f"\n{'=' * 55}")
    print("加载 Expert Iteration 模型...")
    expert_model = load_model(args.expert_checkpoint)
    df_expert, t_expert = infer_all(expert_model, "Expert-Iter")
    Path(args.output_expert_csv).parent.mkdir(parents=True, exist_ok=True)
    df_expert.to_csv(args.output_expert_csv, index=False)
    print(f"  推理时间: {t_expert:.1f}秒 ({t_expert / len(df_expert):.1f}秒/条)")

    print(f"\n{'=' * 55}")
    print("加载 CT Fine-tuned 模型...")
    ft_model = load_model(args.finetune_checkpoint)
    df_ft, t_ft = infer_all(ft_model, "CT-FT")
    Path(args.output_ft_csv).parent.mkdir(parents=True, exist_ok=True)
    df_ft.to_csv(args.output_ft_csv, index=False)
    print(f"  推理时间: {t_ft:.1f}秒 ({t_ft / len(df_ft):.1f}秒/条)")

    print(f"\n{'=' * 55}")
    print(f"读取已有ICR结果: {args.icr_result_csv}")
    df_icr_raw = pd.read_csv(args.icr_result_csv)
    df_icr = pd.DataFrame({
        "protein": df_icr_raw["protein"],
        "CSI": df_icr_raw["icr_CSI"],
        "CFD": df_icr_raw["icr_CFD"],
        "CIS": df_icr_raw["icr_CIS"],
        "MFE": df_icr_raw["icr_MFE"],
    })
    print(f"  共 {len(df_icr)} 条")

    summary_rows = [
        {"method": "CT Fine-tuned", "CSI": df_ft["CSI"].mean(), "CFD": df_ft["CFD"].mean(), "CIS": df_ft["CIS"].mean(), "MFE": df_ft["MFE"].mean(), "time_sec": t_ft},
        {"method": "ICR (Ours)", "CSI": df_icr["CSI"].mean(), "CFD": df_icr["CFD"].mean(), "CIS": df_icr["CIS"].mean(), "MFE": df_icr["MFE"].mean(), "time_sec": None},
        {"method": "Expert Iteration", "CSI": df_expert["CSI"].mean(), "CFD": df_expert["CFD"].mean(), "CIS": df_expert["CIS"].mean(), "MFE": df_expert["MFE"].mean(), "time_sec": t_expert},
    ]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.output_csv, index=False)

    print(f"\n{'=' * 65}")
    print("完整汇总对比（52条benchmark）")
    print(f"{'=' * 65}")
    print(f"{'方法':<25} {'CSI↑':>8} {'CFD↓':>8} {'CIS↓':>8} {'MFE↓':>10} {'推理时间':>10}")
    print("-" * 65)
    print(f"{'CT Fine-tuned':<25} {df_ft['CSI'].mean():>8.4f} {df_ft['CFD'].mean():>8.4f} {df_ft['CIS'].mean():>8.4f} {df_ft['MFE'].mean():>10.2f} {t_ft:>8.1f}s")
    print(f"{'ICR (Ours)':<25} {df_icr['CSI'].mean():>8.4f} {df_icr['CFD'].mean():>8.4f} {df_icr['CIS'].mean():>8.4f} {df_icr['MFE'].mean():>10.2f} {'(ICR搜索)':>10}")
    print(f"{'Expert Iteration':<25} {df_expert['CSI'].mean():>8.4f} {df_expert['CFD'].mean():>8.4f} {df_expert['CIS'].mean():>8.4f} {df_expert['MFE'].mean():>10.2f} {t_expert:>8.1f}s")

    print("\n结果文件:")
    print(f"  {args.output_ft_csv}")
    print(f"  {args.output_expert_csv}")
    print(f"  {args.output_csv}")


if __name__ == "__main__":
    main()
