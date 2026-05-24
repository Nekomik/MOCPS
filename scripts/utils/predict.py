"""
predict.py — command-line MOCPS inference script
用法：
    python predict.py --protein "MALWMRLLP..."
    python predict.py --protein "MALWMRLLP..." --use_mfe
"""

import argparse
import time
import warnings

import pandas as pd

from mocps.config import BASE_MODEL_DIR, DEFAULT_HUMAN_CHECKPOINT, DEFAULT_HUMAN_EVAL_CSV, DEFAULT_ORGANISM
from mocps.sequences import normalize_protein

warnings.filterwarnings("ignore")

# ── 配置 ───────────────────────────────────────────────────────────────────────
BASE_MODEL = BASE_MODEL_DIR
CHECKPOINT = DEFAULT_HUMAN_CHECKPOINT
DATA_CSV = DEFAULT_HUMAN_EVAL_CSV
ORGANISM = DEFAULT_ORGANISM

# ── 工具函数 ───────────────────────────────────────────────────────────────────
def print_banner():
    print()
    print("=" * 58)
    print("   mRNA 密码子多目标优化系统 (MOCPS)")
    print("   基于 CodonTransformer + ICR 推理优化")
    print("=" * 58)

def compute_all_metrics(dna, csi_weights, codon_freq):
    csi = compute_csi(dna, csi_weights)
    cfd = compute_cfd(dna, codon_freq) / 100
    cis = compute_cis(dna)
    mfe = compute_mfe(dna)
    return {"CSI": csi, "CFD": cfd, "CIS": cis, "MFE": mfe}

def print_metrics_comparison(init_m, icr_m):
    print()
    print(f"  {'指标':<14} {'Fine-tuned基线':>14} {'ICR优化后':>12} {'改善':>10}")
    print("  " + "-" * 54)
    for key, higher_better in [('CSI', True), ('CFD', False), ('CIS', False), ('MFE', False)]:
        delta = icr_m[key] - init_m[key]
        flag  = "✅" if (delta >= 0) == higher_better or delta == 0 else "⚠️"
        label = f"{key}{'(kcal/mol)' if key=='MFE' else ''}"
        if key == 'CIS':
            print(f"  {label:<14} {init_m[key]:>14.0f} {icr_m[key]:>12.0f} {delta:>+9.0f} {flag}")
        elif key == 'MFE':
            print(f"  {label:<14} {init_m[key]:>14.1f} {icr_m[key]:>12.1f} {delta:>+9.1f} {flag}")
        else:
            print(f"  {label:<14} {init_m[key]:>14.4f} {icr_m[key]:>12.4f} {delta:>+9.4f} {flag}")
    print()

# ── 主流程 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MOCPS codon optimization inference")
    parser.add_argument("--protein", type=str, required=True, help="氨基酸序列（单字母格式）")
    parser.add_argument("--base_model", type=str, default=str(BASE_MODEL), help="Base CodonTransformer 模型目录")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT), help="fine-tuned checkpoint 路径")
    parser.add_argument("--data_csv", type=str, default=str(DATA_CSV), help="参考评估 CSV")
    parser.add_argument("--organism", type=str, default=ORGANISM, help="目标物种")
    parser.add_argument("--use_mfe", action="store_true", help="开启MFE多候选筛选")
    parser.add_argument("--n_rounds", type=int, default=5, help="ICR迭代轮数")
    args = parser.parse_args()

    print_banner()
    protein = normalize_protein(args.protein)
    print(f"\n  物种：{args.organism}")
    print(f"  蛋白质长度：{len(protein)} 个氨基酸")
    print(f"  序列（前50位）：{protein[:50]}{'...' if len(protein)>50 else ''}")
    print(f"  ICR轮数：{args.n_rounds}，MFE筛选：{'开启' if args.use_mfe else '关闭'}")

    print("  设备：加载中")

    # Step 1：加载模型和参考数据
    print("\n[1/4] 加载模型和参考数据...")
    from CodonTransformer.CodonData import get_codon_frequencies
    from CodonTransformer.CodonEvaluation import get_CSI_weights
    from CodonTransformer.CodonPrediction import predict_dna_sequence
    from mocps.icr import ICROptimizer
    from mocps.metrics import compute_cfd, compute_cis, compute_csi, compute_mfe
    from mocps.modeling import load_tokenizer_and_model
    from mocps.sequences import normalize_dna

    t0 = time.time()
    ref_df = pd.read_csv(args.data_csv)
    ref_dnas   = ref_df['natural_dna'].head(500).tolist()
    codon_freq = get_codon_frequencies(dna_sequences=ref_dnas, organism=args.organism)
    csi_weights = get_CSI_weights(ref_dnas)

    tokenizer, model, device = load_tokenizer_and_model(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        attention_type="original_full",
    )
    print(f"      设备：{device}")
    print(f"      完成（{time.time()-t0:.1f}s）")

    # Step 2：Fine-tuned模型推理，生成初始序列
    print("\n[2/4] Fine-tuned 模型推理，生成初始序列...")
    t1 = time.time()
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
    init_dna     = normalize_dna(result.predicted_dna)
    init_metrics = compute_all_metrics(init_dna, csi_weights, codon_freq)
    print(f"      完成（{time.time()-t1:.1f}s）")
    print(f"      初始序列长度：{len(init_dna)} bp")

    # Step 3：ICR多目标推理优化
    print(f"\n[3/4] ICR 迭代密码子精炼（{args.n_rounds}轮）...")
    t2 = time.time()
    optimizer = ICROptimizer(
        model=model, tokenizer=tokenizer, organism=args.organism,
        codon_frequencies=codon_freq, csi_weights=csi_weights,
        device=device, w_csi=2.0, w_cfd=1.0, w_cis=2.0,
    )
    icr_dna, history = optimizer.optimize(
        protein=protein,
        initial_dna=init_dna,
        n_rounds=args.n_rounds,
        top_k_frac=0.15,
    )
    print(f"      完成（{time.time()-t2:.1f}s）")

    # Step 4：MFE多候选筛选（可选）
    final_dna = icr_dna
    if args.use_mfe:
        print("\n[4/4] MFE 多候选筛选（生成5条候选）...")
        t3 = time.time()
        candidates = [icr_dna]
        for _ in range(4):
            try:
                r = predict_dna_sequence(
                    protein=protein, organism=args.organism, device=device,
                    tokenizer=tokenizer, model=model,
                    attention_type="original_full",
                    deterministic=False, temperature=0.8,
                    top_p=0.95, match_protein=True,
                )
                cand = normalize_dna(r.predicted_dna)
                cand, _ = optimizer.optimize(
                    protein=protein, initial_dna=cand,
                    n_rounds=3, top_k_frac=0.10,
                )
                candidates.append(cand)
            except Exception:
                pass
        mfe_scores = [compute_mfe(c) for c in candidates]
        best_idx   = mfe_scores.index(min(mfe_scores))
        final_dna  = candidates[best_idx]
        print(f"      {len(candidates)}条候选，最优MFE={mfe_scores[best_idx]:.1f} kcal/mol（{time.time()-t3:.1f}s）")
    else:
        print("\n[4/4] 跳过MFE筛选（可加 --use_mfe 开启）")

    # 最终指标
    icr_metrics = compute_all_metrics(final_dna, csi_weights, codon_freq)
    total_time  = time.time() - t0

    # 输出结果
    print()
    print("=" * 58)
    print("   优化结果")
    print("=" * 58)
    print(f"\n  优化后DNA序列（前60bp）：")
    print(f"  {final_dna[:60]}...")
    print(f"  完整序列长度：{len(final_dna)} bp")
    print_metrics_comparison(init_metrics, icr_metrics)
    print("=" * 58)
    print(f"  总耗时：{total_time:.1f} 秒")
    print("=" * 58)
    print()

if __name__ == "__main__":
    main()
