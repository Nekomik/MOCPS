"""
ablation_study.py — 消融实验脚本
四组配置在52条benchmark上对比：
  A: CSI only          (w_csi=2.0, w_cfd=0, w_cis=0, no MFE)
  B: CSI + CFD         (w_csi=2.0, w_cfd=1.0, w_cis=0, no MFE)
  C: CSI + CFD + CIS   (w_csi=2.0, w_cfd=1.0, w_cis=2.0, no MFE)
  D: Full (Ours)       (w_csi=2.0, w_cfd=1.0, w_cis=2.0, use MFE) ← 已有结果，跳过
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from mocps.config import BASE_MODEL_DIR, CHECKPOINTS_DIR, DEFAULT_HUMAN_EVAL_CSV, PROJECT_ROOT, RESULTS_DIR
from mocps.metrics import compute_cfd, compute_cis, compute_csi, compute_mfe
from mocps.sequences import normalize_dna, normalize_protein

ORGANISM = "Homo sapiens"
ABLATION_CONFIGS = [
    {"name": "CSI only", "w_csi": 2.0, "w_cfd": 0.0, "w_cis": 0.0, "use_mfe": False},
    {"name": "CSI + CFD", "w_csi": 2.0, "w_cfd": 1.0, "w_cis": 0.0, "use_mfe": False},
    {"name": "CSI + CFD + CIS", "w_csi": 2.0, "w_cfd": 1.0, "w_cis": 2.0, "use_mfe": False},
]
FULL_RESULT = {
    "name": "Full (Ours)",
    "w_csi": 2.0, "w_cfd": 1.0, "w_cis": 2.0, "use_mfe": True,
    "CSI": 0.9743, "CFD": 0.0002, "CIS": 0.0385, "MFE": -499.28,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run ICR ablation study on the 52-protein benchmark")
    parser.add_argument(
        "--xlsx",
        default=str(PROJECT_ROOT / "data" / "Fig. 4, Supplementary Fig. 19f-h, 20, 21, 22 and 25.xlsx"),
        help="Benchmark Excel file path",
    )
    parser.add_argument("--sheet", default="DNA sequences, CSI, CIS-element", help="Excel sheet name")
    parser.add_argument("--ref_csv", default=str(DEFAULT_HUMAN_EVAL_CSV), help="Reference evaluation CSV")
    parser.add_argument("--base_model", default=str(BASE_MODEL_DIR), help="Base CodonTransformer model directory")
    parser.add_argument("--checkpoint", default=str(CHECKPOINTS_DIR / "homo_sapiens" / "finetune.ckpt"))
    parser.add_argument("--output_dir", default=str(RESULTS_DIR), help="Directory for ablation result CSVs")
    parser.add_argument("--organism", default=ORGANISM, help="Target organism")
    parser.add_argument("--n_rounds", type=int, default=5)
    parser.add_argument("--top_k_frac", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()

    from CodonTransformer.CodonData import get_codon_frequencies
    from CodonTransformer.CodonEvaluation import get_CSI_weights
    from CodonTransformer.CodonPrediction import predict_dna_sequence
    from mocps.icr import ICROptimizer
    from mocps.modeling import load_tokenizer_and_model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载数据和模型...")
    benchmark = pd.read_excel(args.xlsx, sheet_name=args.sheet)
    human_bm = benchmark[benchmark["organism"] == args.organism].copy().reset_index(drop=True)
    print(f"  benchmark: {len(human_bm)} 条")

    ref_df = pd.read_csv(args.ref_csv)
    ref_dnas = ref_df["natural_dna"].head(500).tolist()
    codon_freq = get_codon_frequencies(dna_sequences=ref_dnas, organism=args.organism)
    csi_weights = get_CSI_weights(ref_dnas)

    tokenizer, model, device = load_tokenizer_and_model(
        args.base_model,
        checkpoint=args.checkpoint,
        attention_type="original_full",
    )
    print(f"  设备: {device}")
    print("  模型加载完成")

    all_results = {}
    for config in ABLATION_CONFIGS:
        name = config["name"]
        print(f"\n{'=' * 60}")
        print(f"配置: {name}")
        print(f"  w_csi={config['w_csi']}, w_cfd={config['w_cfd']}, w_cis={config['w_cis']}, MFE筛选=否")
        print(f"{'=' * 60}")

        optimizer = ICROptimizer(
            model=model,
            tokenizer=tokenizer,
            organism=args.organism,
            codon_frequencies=codon_freq,
            csi_weights=csi_weights,
            device=device,
            w_csi=config["w_csi"],
            w_cfd=config["w_cfd"],
            w_cis=config["w_cis"],
        )

        results = []
        t_start = time.time()

        for idx, row in human_bm.iterrows():
            protein = normalize_protein(str(row["protein sequence"]))
            print(f"  [{idx + 1}/{len(human_bm)}] {row['protein']} (len={len(protein)})", end="", flush=True)

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
                init_dna = normalize_dna(result.predicted_dna)
                icr_dna, _ = optimizer.optimize(
                    protein=protein,
                    initial_dna=init_dna,
                    n_rounds=args.n_rounds,
                    top_k_frac=args.top_k_frac,
                )

                csi = compute_csi(icr_dna, csi_weights)
                cfd = compute_cfd(icr_dna, codon_freq)
                cis = compute_cis(icr_dna)
                mfe = compute_mfe(icr_dna)

                results.append({"protein": row["protein"], "CSI": csi, "CFD": cfd, "CIS": cis, "MFE": mfe})
                print(f"  CSI={csi:.4f} CIS={cis} MFE={mfe:.1f}")
            except Exception as exc:
                print(f"  错误: {exc}")

        elapsed = time.time() - t_start
        df_res = pd.DataFrame(results)
        out_path = output_dir / f"ablation_{name.replace(' ', '_').replace('+', 'p')}.csv"
        df_res.to_csv(out_path, index=False)

        print(f"\n  [{name}] 完成，耗时{elapsed / 60:.1f}分钟")
        print(f"  均值 → CSI={df_res['CSI'].mean():.4f}  CFD={df_res['CFD'].mean():.4f}  CIS={df_res['CIS'].mean():.4f}  MFE={df_res['MFE'].mean():.1f}")
        print(f"  结果保存至: {out_path}")
        all_results[name] = df_res

    print(f"\n{'=' * 60}")
    print("消融实验汇总（含配置D已有结果）")
    print(f"{'=' * 60}")
    print(f"{'配置':<22} {'CSI↑':>8} {'CFD↓':>8} {'CIS↓':>8} {'MFE↓':>10}")
    print("-" * 60)

    summary = []
    for config in ABLATION_CONFIGS:
        name = config["name"]
        df = all_results[name]
        print(f"{name:<22} {df['CSI'].mean():>8.4f} {df['CFD'].mean():>8.4f} {df['CIS'].mean():>8.4f} {df['MFE'].mean():>10.1f}")
        summary.append({
            "配置": name,
            "w_csi": config["w_csi"], "w_cfd": config["w_cfd"],
            "w_cis": config["w_cis"], "use_mfe": config["use_mfe"],
            "CSI": df["CSI"].mean(), "CFD": df["CFD"].mean(),
            "CIS": df["CIS"].mean(), "MFE": df["MFE"].mean(),
        })

    d = FULL_RESULT
    print(f"{'Full (Ours)':<22} {d['CSI']:>8.4f} {d['CFD']:>8.4f} {d['CIS']:>8.4f} {d['MFE']:>10.1f}")
    summary.append({
        "配置": d["name"],
        "w_csi": d["w_csi"], "w_cfd": d["w_cfd"],
        "w_cis": d["w_cis"], "use_mfe": d["use_mfe"],
        "CSI": d["CSI"], "CFD": d["CFD"], "CIS": d["CIS"], "MFE": d["MFE"],
    })

    summary_path = output_dir / "ablation_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"\n汇总结果保存至: {summary_path}")


if __name__ == "__main__":
    main()
