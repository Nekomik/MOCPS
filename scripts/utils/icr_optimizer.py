"""
icr_optimizer.py
================
迭代密码子精炼（Iterative Codon Refinement, ICR）
基于 CodonTransformer CSI fine-tuned 模型，在推理阶段对生成序列进行多目标优化。

不改模型参数，只在序列空间里搜索更好的密码子组合。

用法：
    python icr_optimizer.py \
        --csv        artifacts/results/all_eval_with_my_finetune/homo_sapiens_all_eval_with_my_finetune.csv \
        --checkpoint artifacts/checkpoints/homo_sapiens/finetune.ckpt \
        --base_model /root/autodl-tmp/CodonTransformer_base \
        --organism   "Homo sapiens" \
        --output_csv artifacts/results/homo_sapiens_icr_results.csv \
        --n_samples  200 \
        --icr_rounds 5 \
        --top_k_frac 0.15
"""

import argparse
from pathlib import Path

import pandas as pd

from mocps.config import BASE_MODEL_DIR, DEFAULT_HUMAN_CHECKPOINT, DEFAULT_HUMAN_EVAL_CSV, DEFAULT_ORGANISM, RESULTS_DIR
from mocps.sequences import normalize_protein


# ─────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_HUMAN_EVAL_CSV))
    parser.add_argument("--checkpoint", default=str(DEFAULT_HUMAN_CHECKPOINT))
    parser.add_argument("--base_model", default=str(BASE_MODEL_DIR))
    parser.add_argument("--organism", default=DEFAULT_ORGANISM)
    parser.add_argument("--output_csv", default=str(RESULTS_DIR / "homo_sapiens_icr_results.csv"))
    parser.add_argument("--n_samples",   type=int,   default=200,  help="评估序列数量")
    parser.add_argument("--icr_rounds",  type=int,   default=5)
    parser.add_argument("--top_k_frac",  type=float, default=0.15)
    parser.add_argument("--use_mfe",     action="store_true", help="是否用MFE做最终筛选")
    args = parser.parse_args()

    from CodonTransformer.CodonData import get_codon_frequencies
    from CodonTransformer.CodonEvaluation import get_CSI_weights
    from CodonTransformer.CodonPrediction import predict_dna_sequence
    from mocps.icr import ICROptimizer
    from mocps.metrics import compute_gc, compute_mfe
    from mocps.modeling import load_tokenizer_and_model
    from mocps.sequences import normalize_dna

    print("[ICR] 设备: 加载中")

    # ── 1. 读取数据 ──────────────────────────────────────────────
    print(f"[ICR] 读取数据: {args.csv}")
    df = pd.read_csv(args.csv)

    # 只取有效的已推演序列
    valid_mask = df["my_translated_match"] == True
    df_valid = df[valid_mask].head(args.n_samples).copy().reset_index(drop=True)
    print(f"[ICR] 有效序列: {len(df_valid)} 条")

    # ── 2. 构建评估工具 ──────────────────────────────────────────
    print("[ICR] 构建 codon_frequencies 和 CSI weights...")
    ref_dnas = df["natural_dna"].head(500).tolist()
    codon_freq = get_codon_frequencies(
        dna_sequences=ref_dnas,
        organism=args.organism,
    )
    csi_weights = get_CSI_weights(ref_dnas)

    # ── 3. 加载模型 ──────────────────────────────────────────────
    print(f"[ICR] 加载模型...")
    tokenizer, model, device = load_tokenizer_and_model(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        attention_type="original_full",
    )
    print(f"[ICR] 设备: {device}")

    # ── 4. 初始化 ICR ────────────────────────────────────────────
    optimizer = ICROptimizer(
        model=model,
        tokenizer=tokenizer,
        organism=args.organism,
        codon_frequencies=codon_freq,
        csi_weights=csi_weights,
        device=device,
        w_csi=1.0,
        w_cfd=2.0,
        w_cis=0.5,
    )

    # ── 5. 对每条序列做 ICR ──────────────────────────────────────
    results = []

    for idx, row in df_valid.iterrows():
        protein    = normalize_protein(row["protein"])
        init_dna   = normalize_dna(row["finetune_dna"])  # CSI fine-tuned 生成的序列

        if len(protein) > 2040 or len(init_dna) < 3:
            print(f"[{idx+1}/{len(df_valid)}] 跳过（序列过长或无效）")
            continue

        print(f"\n[{idx+1}/{len(df_valid)}] 优化中... (蛋白质长度={len(protein)})")

        # 记录初始指标
        init_reward, init_metrics = optimizer._compute_reward(init_dna)
        print(
            f"    初始: reward={init_reward:+.4f}  "
            f"CSI={init_metrics['CSI']:.4f}  "
            f"CFD={init_metrics['CFD']:.4f}  "
            f"CIS={init_metrics['CIS']}"
        )

        # ICR 优化
        icr_dna, history = optimizer.optimize(
            protein=protein,
            initial_dna=init_dna,
            n_rounds=args.icr_rounds,
            top_k_frac=args.top_k_frac,
        )

        # MFE 推理筛选：生成5条候选，选MFE最低的
        init_mfe = compute_mfe(init_dna)
        candidates = [icr_dna]
        try:
            from CodonTransformer.CodonPrediction import predict_dna_sequence
            for _ in range(4):
                if len(protein) <= 2040:
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
                    cand_dna = normalize_dna(result.predicted_dna)
                    # 对候选也做ICR优化
                    cand_dna, _ = optimizer.optimize(
                        protein=protein,
                        initial_dna=cand_dna,
                        n_rounds=3,
                        top_k_frac=0.10,
                    )
                    candidates.append(cand_dna)
        except Exception:
            pass

        # 选MFE最低的候选
        mfe_scores = [compute_mfe(c) for c in candidates]
        best_idx   = mfe_scores.index(min(mfe_scores))
        icr_dna    = candidates[best_idx]
        icr_mfe    = mfe_scores[best_idx]
        print(f"    MFE筛选: {len(candidates)}条候选, 最优MFE={icr_mfe:.2f}")

        # 记录结果
        final_reward, final_metrics = optimizer._compute_reward(icr_dna)
        results.append({
            "protein_id":   idx,
            "organism":     args.organism,
            "protein":      protein,

            # 初始序列（CSI fine-tuned）
            "init_dna":     init_dna,
            "init_CSI":     init_metrics["CSI"],
            "init_CFD":     init_metrics["CFD"],
            "init_CIS":     init_metrics["CIS"],
            "init_MFE":     init_mfe,
            "init_GC":      compute_gc(init_dna),

            # ICR 优化后
            "icr_dna":      icr_dna,
            "icr_CSI":      final_metrics["CSI"],
            "icr_CFD":      final_metrics["CFD"],
            "icr_CIS":      final_metrics["CIS"],
            "icr_MFE":      icr_mfe,
            "icr_GC":       compute_gc(icr_dna),

            # 改进量
            "delta_CSI":    final_metrics["CSI"] - init_metrics["CSI"],
            "delta_CFD":    final_metrics["CFD"] - init_metrics["CFD"],
            "delta_CIS":    final_metrics["CIS"] - init_metrics["CIS"],
            "delta_MFE":    icr_mfe - init_mfe,
            "delta_reward": final_reward - init_reward,

            # 原始数据（用于对比）
            "natural_CSI":  row.get("natural_CSI", None),
            "base_CSI":     row.get("base_CSI", None),
            "finetune_CSI": row.get("finetune_CSI", None),
        })

    # ── 6. 保存结果 ──────────────────────────────────────────────
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n[ICR] 结果已保存: {args.output_csv}")

    # ── 7. 打印汇总 ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("汇总对比（均值）")
    print("="*60)
    print(f"{'指标':<12} {'初始(CSI FT)':>14} {'ICR优化后':>12} {'改进':>10}")
    print("-"*60)
    for metric, col_init, col_icr in [
        ("CSI ↑",  "init_CSI", "icr_CSI"),
        ("CFD ↓",  "init_CFD", "icr_CFD"),
        ("CIS ↓",  "init_CIS", "icr_CIS"),
        ("MFE ↓",  "init_MFE", "icr_MFE"),
        ("GC%",    "init_GC",  "icr_GC"),
    ]:
        v_init = results_df[col_init].mean()
        v_icr  = results_df[col_icr].mean()
        delta  = v_icr - v_init
        arrow  = "✅" if (
            (metric.endswith("↑") and delta > 0) or
            (metric.endswith("↓") and delta < 0)
        ) else "❌"
        print(f"{metric:<12} {v_init:>14.4f} {v_icr:>12.4f} {delta:>+10.4f} {arrow}")
    print("="*60)


if __name__ == "__main__":
    main()
