"""
calc_commercial_and_icr_time.py
1. 计算商业工具（Twist/IDT/Genewiz）的CFD和MFE
2. 提示如何统计ICR在52条benchmark上的推理时间
"""

import argparse
from pathlib import Path

import pandas as pd

from mocps.config import DEFAULT_HUMAN_EVAL_CSV, PROJECT_ROOT, RESULTS_DIR
from mocps.metrics import compute_cis, compute_mfe
from mocps.sequences import normalize_dna

ORGANISM = "Homo sapiens"


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate commercial-tool metrics and summarize ICR timing guidance")
    parser.add_argument(
        "--xlsx",
        default=str(PROJECT_ROOT / "data" / "Fig. 4, Supplementary Fig. 19f-h, 20, 21, 22 and 25.xlsx"),
        help="Benchmark Excel file path",
    )
    parser.add_argument("--sheet", default="DNA sequences, CSI, CIS-element", help="Excel sheet name")
    parser.add_argument("--ref_csv", default=str(DEFAULT_HUMAN_EVAL_CSV), help="Reference evaluation CSV")
    parser.add_argument("--icr_csv", default=str(RESULTS_DIR / "homo_sapiens_benchmark_icr.csv"))
    parser.add_argument("--output_csv", default=str(RESULTS_DIR / "commercial_cfd_mfe.csv"))
    parser.add_argument("--organism", default=ORGANISM, help="Target organism")
    return parser.parse_args()


def main():
    args = parse_args()

    from CodonTransformer.CodonData import get_codon_frequencies
    from CodonTransformer.CodonEvaluation import get_cfd

    print("构建评估工具...")
    ref_df = pd.read_csv(args.ref_csv)
    ref_dnas = ref_df["natural_dna"].head(500).tolist()
    codon_freq = get_codon_frequencies(dna_sequences=ref_dnas, organism=args.organism)

    print("读取benchmark数据...")
    benchmark = pd.read_excel(args.xlsx, sheet_name=args.sheet)
    human_bm = benchmark[benchmark["organism"] == args.organism].copy().reset_index(drop=True)
    print(f"  共 {len(human_bm)} 条")

    methods = [
        ("Twist", "twist_DNA", "twist_CSI", "twist_CIS"),
        ("IDT", "idt_dna", "idt_CSI", "idt_CIS"),
        ("Genewiz", "genewiz_dna", "genewiz_CSI", "genewiz_CIS"),
    ]

    results = []
    for method_name, dna_col, csi_col, cis_col in methods:
        print(f"\n计算 {method_name}...")
        cfd_list, mfe_list, csi_list, cis_list = [], [], [], []

        for idx, row in human_bm.iterrows():
            try:
                dna = normalize_dna(str(row[dna_col]))
                if not dna or dna == "NAN" or len(dna) < 3:
                    continue

                cfd_val = get_cfd(dna.upper(), codon_freq) / 100
                mfe_val = compute_mfe(dna)
                csi_val = float(row[csi_col])
                cis_val = float(row[cis_col]) if cis_col in row and pd.notna(row[cis_col]) else compute_cis(dna)

                cfd_list.append(cfd_val)
                mfe_list.append(mfe_val)
                csi_list.append(csi_val)
                cis_list.append(cis_val)

                print(f"  [{idx + 1}/{len(human_bm)}] CFD={cfd_val:.4f} MFE={mfe_val:.1f}")

            except Exception as exc:
                print(f"  [{idx + 1}] 跳过: {exc}")

        results.append({
            "方法": method_name,
            "CSI均值": round(sum(csi_list) / len(csi_list), 4) if csi_list else None,
            "CFD均值": round(sum(cfd_list) / len(cfd_list), 4) if cfd_list else None,
            "CIS均值": round(sum(cis_list) / len(cis_list), 4) if cis_list else None,
            "MFE均值": round(sum(mfe_list) / len(mfe_list), 2) if mfe_list else None,
            "样本数": len(cfd_list),
        })
        print(
            f"  {method_name}: CSI={results[-1]['CSI均值']} CFD={results[-1]['CFD均值']} "
            f"CIS={results[-1]['CIS均值']} MFE={results[-1]['MFE均值']}"
        )

    print(f"\n读取ICR结果: {args.icr_csv}")
    df_icr = pd.read_csv(args.icr_csv)
    print(f"  共 {len(df_icr)} 条")

    print("\n正在从日志估算ICR推理时间...")
    print("  请查看 artifacts/logs/ 目录下的日志文件，找到ICR跑52条的总耗时")
    print("  或者运行 time_icr_benchmark.py 重新计时")

    df_results = pd.DataFrame(results)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)

    print(f"\n{'=' * 65}")
    print("商业工具完整指标汇总")
    print(f"{'=' * 65}")
    print(f"{'方法':<12} {'CSI↑':>8} {'CFD↓':>8} {'CIS↓':>8} {'MFE↓':>10}")
    print("-" * 50)
    for _, row in df_results.iterrows():
        print(
            f"{row['方法']:<12} {row['CSI均值']:>8.4f} {row['CFD均值']:>8.4f} "
            f"{row['CIS均值']:>8.4f} {row['MFE均值']:>10.2f}"
        )

    print(f"\n结果保存至: {output_csv}")
    print("\n** Use the above values in the benchmark result tables **")


if __name__ == "__main__":
    main()
