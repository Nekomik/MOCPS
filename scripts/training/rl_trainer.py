"""
rl_trainer.py
=============
基于 REINFORCE 的多目标密码子优化 RL 训练脚本。

用法：
    python rl_trainer.py \
        --csv        artifacts/results/all_eval_with_my_finetune/homo_sapiens_all_eval_with_my_finetune.csv \
        --checkpoint artifacts/checkpoints/homo_sapiens/finetune.ckpt \
        --base_model /root/autodl-tmp/CodonTransformer_base \
        --organism   "Homo sapiens" \
        --output_dir artifacts/checkpoints/homo_sapiens_rl \
        --steps      3000 \
        --batch_size 8 \
        --lr         1e-5 \
        --lambda_mlm 0.1 \
        --w_csi      1.0 \
        --w_cfd      1.0 \
        --w_cousin   1.0 \
        --w_cis      1.0
"""

import argparse
import math
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from mocps.config import BASE_MODEL_DIR, CHECKPOINTS_DIR, DEFAULT_HUMAN_EVAL_CSV, DEFAULT_ORGANISM
from mocps.sequences import normalize_dna, normalize_protein


# ─────────────────────────────────────────────────────────────────
# 带 log_prob 的序列采样
# ─────────────────────────────────────────────────────────────────

def get_token_ids_for_organism(organism: str, tokenizer) -> dict:
    """
    构建该物种的 amino_UNK token id 映射。
    参考 CodonTransformer 的 STREAM tokenization 策略。
    """
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY_")
    unk_tokens = {aa: f"{aa}_UNK" for aa in amino_acids}
    token_ids = {}
    for aa, tok in unk_tokens.items():
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if ids:
            token_ids[aa] = ids[0]
    return token_ids


def build_aa_to_codon_tokens(tokenizer) -> dict:
    """
    从 tokenizer 词表里构建 {aa: [(token_str, token_id), ...]} 映射。
    格式: A_GCC, M_ATG 等（长度5，第1位是氨基酸，第2位是_，后三位是密码子）
    只构建一次，在训练开始时调用。
    """
    vocab = tokenizer.get_vocab()
    aa_to_codon_tokens = defaultdict(list)
    for token_str, token_id in vocab.items():
        if (len(token_str) == 5 and token_str[1] == "_"
                and token_str[0].isalpha()
                and all(c in 'atgc' for c in token_str[2:])):  # 只保留真正的 codon
            aa = token_str[0].upper()
            aa_to_codon_tokens[aa].append((token_str, token_id))
    return aa_to_codon_tokens


def score_dna_with_logprob(
    model,
    tokenizer,
    protein: str,
    dna: str,
    organism_id: int,
    device: torch.device,
    aa_to_codon_tokens: dict,
) -> torch.Tensor:
    """
    Step 2：把已生成的 DNA 序列输入模型，计算每个位置的 log_prob 之和。
    这个值可以反向传播（不用 no_grad），用于 REINFORCE loss。

    原理：把 AA_codon token 序列输入模型，对每个位置取对应 codon token 的概率，
    求 log 并求和得到整条序列的 log probability。
    """
    dna = dna.upper()
    # 构建 AA_codon token 序列（已知 codon 的版本，用于打分）
    tokens = []
    for i, aa in enumerate(protein):
        codon = dna[i*3:(i+1)*3] if i*3+3 <= len(dna) else None
        if codon and len(codon) == 3:
            tok = f"{aa}_{codon}"
        else:
            tok = f"{aa}_UNK"
        tokens.append(tok)

    token_str = " ".join(tokens)
    encoding = tokenizer(
        token_str,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )

    input_ids_full = encoding["input_ids"].to(device)          # [1, seq_len]
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = torch.full_like(input_ids_full, fill_value=organism_id).to(device)

    # 把所有 AA_codon token 替换成 AA_UNK（模拟推理输入）
    # 这样模型看到的是"待预测"状态，输出的 logits 才是有意义的预测概率
    input_ids_masked = input_ids_full.clone()
    for i, aa in enumerate(protein):
        seq_pos = i + 1  # 跳过 CLS
        if seq_pos >= input_ids_masked.shape[1]:
            break
        unk_tok = f"{aa.lower()}_unk"
        unk_ids = tokenizer.encode(unk_tok, add_special_tokens=False)
        if unk_ids:
            input_ids_masked[0, seq_pos] = unk_ids[0]

    # 前向传播（保留梯度）
    outputs = model(
        input_ids=input_ids_masked,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    logits = outputs.logits  # [1, seq_len, vocab_size]

    # 对每个位置取对应 codon token 的 log_prob
    log_prob_terms = []

    for i, aa in enumerate(protein):
        seq_pos = i + 1
        if seq_pos >= logits.shape[1]:
            break

        codon = dna[i*3:(i+1)*3] if i*3+3 <= len(dna) else None
        if not codon or len(codon) != 3:
            continue

        # 找该氨基酸所有候选 codon token
        candidates = aa_to_codon_tokens.get(aa, [])
        if not candidates:
            continue

        candidate_ids    = [tid for _, tid in candidates]
        candidate_logits = logits[0, seq_pos, candidate_ids]
        log_probs_cand   = F.log_softmax(candidate_logits, dim=-1)

        # 找当前 codon 对应的 token index
        target_tok = f"{aa.lower()}_{codon.lower()}"
        target_idx = next(
            (j for j, (ts, _) in enumerate(candidates) if ts == target_tok), None
        )
        if target_idx is not None:
            log_prob_terms.append(log_probs_cand[target_idx])

    if not log_prob_terms:
        return torch.zeros(1, device=device, requires_grad=True)
    # 除以序列长度归一化，避免长序列 log_prob 量级过大
    return (torch.stack(log_prob_terms).sum() / len(log_prob_terms)).unsqueeze(0)


def sample_sequence_with_logprob(
    model,
    tokenizer,
    protein: str,
    organism: str,
    organism_id: int,
    device: torch.device,
    aa_to_codon_tokens: dict,
    temperature: float = 0.8,
) -> tuple[str, torch.Tensor]:
    """
    两步走：
    Step 1: 用 predict_dna_sequence(deterministic=False) 采样高质量序列
    Step 2: 用 score_dna_with_logprob 对生成序列打分，得到可反传的 log_prob

    Returns:
        dna: 生成的 DNA 序列（质量有保证）
        log_prob: 可反传的 log probability tensor
    """
    from CodonTransformer.CodonPrediction import predict_dna_sequence

    # Step 1：原生采样（质量有保证，用 temperature 控制随机性）
    model.eval()
    with torch.no_grad():
        result = predict_dna_sequence(
            protein=protein,
            organism=organism,
            device=device,
            tokenizer=tokenizer,
            model=model,
            attention_type="original_full",
            deterministic=False,
            temperature=temperature,
            top_p=0.95,
            match_protein=True,
        )
    dna = normalize_dna(result.predicted_dna)

    # Step 2：对生成序列打分（开启梯度）
    model.train()
    log_prob = score_dna_with_logprob(
        model, tokenizer, protein, dna, organism_id, device, aa_to_codon_tokens
    )

    return dna, log_prob


# ─────────────────────────────────────────────────────────────────
# MLM Loss（防遗忘正则化）
# ─────────────────────────────────────────────────────────────────

def compute_mlm_loss(
    model,
    tokenizer,
    batch_proteins: list[str],
    batch_dnas: list[str],
    organism_id: int,
    device: torch.device,
    mask_prob: float = 0.15,
) -> torch.Tensor:
    """
    计算 MLM loss，用于防止模型在 RL 训练中遗忘原有的 codon 知识。
    使用天然 DNA 序列作为标签。
    """
    if not batch_dnas:
        return torch.tensor(0.0, requires_grad=True, device=device)

    # 构建 AA_codon token 序列
    all_input_ids, all_labels, all_type_ids = [], [], []

    for protein, dna in zip(batch_proteins, batch_dnas):
        tokens = []
        for i, aa in enumerate(protein):
            codon = dna[i*3:(i+1)*3] if i*3+3 <= len(dna) else None
            if codon:
                tok = f"{aa}_{codon}"
            else:
                tok = f"{aa}_UNK"
            tokens.append(tok)

        token_str = " ".join(tokens)
        enc = tokenizer(
            token_str,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].squeeze(0)
        labels    = input_ids.clone()

        # 随机 mask 15% 的位置
        mask = torch.rand(input_ids.shape) < mask_prob
        # 不 mask special tokens (CLS=0, SEP=2, PAD=1)
        special = (input_ids == tokenizer.cls_token_id) | \
                  (input_ids == tokenizer.sep_token_id) | \
                  (input_ids == tokenizer.pad_token_id)
        mask = mask & ~special

        input_ids[mask] = tokenizer.mask_token_id
        labels[~mask]   = -100  # 只对 mask 位置计算 loss

        type_ids = torch.full_like(input_ids, fill_value=organism_id)

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_type_ids.append(type_ids)

    input_ids_batch  = torch.stack(all_input_ids).to(device)
    labels_batch     = torch.stack(all_labels).to(device)
    type_ids_batch   = torch.stack(all_type_ids).to(device)
    attn_mask        = (input_ids_batch != tokenizer.pad_token_id).long()

    outputs = model(
        input_ids=input_ids_batch,
        attention_mask=attn_mask,
        token_type_ids=type_ids_batch,
        labels=labels_batch,
    )
    return outputs.loss


# ─────────────────────────────────────────────────────────────────
# 训练主函数
# ─────────────────────────────────────────────────────────────────

def train(args):
    from CodonTransformer.CodonUtils import ORGANISM2ID
    from mocps.modeling import load_tokenizer_and_model
    from mocps.reward import RewardCalculator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RL] 使用设备: {device}")

    # ── 1. 读取数据 ──────────────────────────────────────────────
    print(f"[RL] 读取数据: {args.csv}")
    df = pd.read_csv(args.csv)
    df["protein"]    = df["protein"].map(normalize_protein)
    df["natural_dna"] = df["natural_dna"].map(normalize_dna)

    # 80% 训练，20% 验证
    n_train = int(len(df) * 0.8)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df   = df.iloc[n_train:].reset_index(drop=True)
    print(f"[RL] 训练集: {len(train_df)} 条，验证集: {len(val_df)} 条")

    # ── 2. 构建 codon_frequencies 和 RewardCalculator ────────────
    print("[RL] 构建 codon_frequencies（使用前 500 条天然序列）...")
    from CodonTransformer.CodonData import get_codon_frequencies
    ref_dnas = train_df["natural_dna"].head(500).tolist()
    codon_freq = get_codon_frequencies(
        dna_sequences=ref_dnas,
        organism=args.organism,
    )

    calc = RewardCalculator(
        natural_dnas=ref_dnas,
        codon_frequencies=codon_freq,
        w_csi=args.w_csi,
        w_cfd=args.w_cfd,
        w_cousin=args.w_cousin,
        w_cis=args.w_cis,
    )

    # 首轮校准 CIS/COUSIN 归一化统计（用训练集 finetune_dna）
    print("[RL] 校准归一化统计数据...")
    calib_dnas = train_df["finetune_dna"].head(200).tolist()
    calc.calibrate_norm_stats(calib_dnas, n_samples=200)

    # ── 3. 加载模型 ──────────────────────────────────────────────
    print(f"[RL] 加载模型: {args.base_model}")
    tokenizer, model, device = load_tokenizer_and_model(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        attention_type="original_full",
        eval_mode=False,
    )

    # ── 4. 获取 organism_id ──────────────────────────────────────
    organism_id = ORGANISM2ID.get(args.organism, 0)
    print(f"[RL] organism_id: {organism_id} ({args.organism})")

    print(f"[RL] 构建 aa_to_codon_tokens 映射...")
    aa_to_codon_tokens = build_aa_to_codon_tokens(tokenizer)
    print(f"[RL] 共 {len(aa_to_codon_tokens)} 种氨基酸的 codon token 映射")

    # ── 5. 优化器 ────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # ── 6. 输出目录 ──────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.csv"

    # 训练日志
    log_records = []
    best_reward = -float("inf")

    # ── 7. 训练循环 ──────────────────────────────────────────────
    train_list = train_df[["protein", "natural_dna"]].values.tolist()
    step = 0

    print(f"\n[RL] 开始训练，共 {args.steps} 步，batch_size={args.batch_size}")
    print("="*70)

    while step < args.steps:
        # 随机采样一个 batch
        batch = random.sample(train_list, min(args.batch_size, len(train_list)))
        batch_proteins = [b[0] for b in batch]
        batch_nat_dnas = [b[1] for b in batch]

        # ── Step A：采样生成序列 + 记录 log_prob ─────────────────
        gen_dnas, log_probs = [], []
        for protein in batch_proteins:
            # 跳过过长序列（模型最大支持 2046 个 token，每个氨基酸对应1个token）
            if len(protein) > 2040:
                gen_dnas.append("")
                log_probs.append(torch.zeros(1, device=device))
                continue
            try:
                dna, lp = sample_sequence_with_logprob(
                    model, tokenizer, protein, args.organism,
                    organism_id, device, aa_to_codon_tokens, temperature=0.8,
                )
                gen_dnas.append(dna)
                log_probs.append(lp)
            except Exception as e:
                gen_dnas.append("")
                log_probs.append(torch.zeros(1, device=device))

        # ── Step B：计算 reward ──────────────────────────────────
        rewards, metrics_list = calc.compute_batch(gen_dnas)

        # 过滤 nan（序列生成失败时可能出现）
        valid = [
            i for i, (r, dna) in enumerate(zip(rewards, gen_dnas))
            if not math.isnan(r) and len(dna) >= 3
        ]
        if len(valid) < 2:
            print(f"[Step {step}] 有效序列不足，跳过")
            step += 1
            continue

        rewards_v = torch.tensor([rewards[i] for i in valid], dtype=torch.float32).to(device)

        # ── Step C：REINFORCE loss ───────────────────────────────
        baseline   = rewards_v.mean()
        advantages = rewards_v - baseline
        # 归一化 advantages，稳定训练
        if advantages.std() > 1e-6:
            advantages = advantages / (advantages.std() + 1e-8)

        # log_probs 现在是 tensor（可反传），stack 后计算 loss
        log_probs_tensor = torch.cat([log_probs[i].to(device) for i in valid])
        rl_loss = -(log_probs_tensor * advantages.detach()).mean()

        # ── Step D：MLM 正则化 loss ──────────────────────────────
        mlm_proteins = [batch_proteins[i] for i in valid]
        mlm_nat_dnas = [batch_nat_dnas[i]  for i in valid]
        mlm_loss = compute_mlm_loss(
            model, tokenizer, mlm_proteins, mlm_nat_dnas,
            organism_id, device,
        )

        total_loss = rl_loss + args.lambda_mlm * mlm_loss

        # ── Step E：反传 ─────────────────────────────────────────
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step += 1

        # ── 日志 ─────────────────────────────────────────────────
        if step % 10 == 0:
            avg_reward = rewards_v.mean().item()
            avg_csi    = sum(metrics_list[i]["CSI"]    for i in valid) / len(valid)
            avg_cfd    = sum(metrics_list[i]["CFD"]    for i in valid) / len(valid)
            avg_cousin = sum(metrics_list[i]["COUSIN"] for i in valid) / len(valid)
            avg_cis    = sum(metrics_list[i]["CIS"]    for i in valid) / len(valid)

            print(
                f"[Step {step:4d}] "
                f"reward={avg_reward:+.4f}  "
                f"CSI={avg_csi:.4f}  "
                f"CFD={avg_cfd:.4f}  "
                f"COUSIN={avg_cousin:.4f}  "
                f"CIS={avg_cis:.2f}  "
                f"rl_loss={rl_loss.item():.4f}  "
                f"mlm_loss={mlm_loss.item():.4f}"
            )

            log_records.append({
                "step": step,
                "reward": avg_reward,
                "CSI": avg_csi,
                "CFD": avg_cfd,
                "COUSIN": avg_cousin,
                "CIS": avg_cis,
                "rl_loss": rl_loss.item(),
                "mlm_loss": mlm_loss.item(),
            })

        # ── 验证 + 保存 checkpoint ────────────────────────────────
        if step % 200 == 0:
            print(f"\n[Step {step}] 验证中...")
            val_rewards = []
            val_sample = val_df.sample(min(50, len(val_df)), random_state=step)
            for _, row in val_sample.iterrows():
                if len(row["protein"]) > 2040:
                    continue
                try:
                    dna, _ = sample_sequence_with_logprob(
                        model, tokenizer, row["protein"], args.organism,
                        organism_id, device, aa_to_codon_tokens, temperature=0.8,
                    )
                    r, _ = calc.compute(dna)
                    if not math.isnan(r):
                        val_rewards.append(r)
                except Exception:
                    continue

            val_reward = sum(val_rewards) / len(val_rewards) if val_rewards else -999

            ckpt_path = output_dir / f"rl_step{step}.ckpt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Step {step}] 验证 reward={val_reward:.4f}  已保存: {ckpt_path}")

            if val_reward > best_reward:
                best_reward = val_reward
                best_path   = output_dir / "rl_best.ckpt"
                torch.save(model.state_dict(), best_path)
                print(f"[Step {step}] ★ 新最优 checkpoint: {best_path}")

            print()

    # ── 8. 保存训练日志 ──────────────────────────────────────────
    pd.DataFrame(log_records).to_csv(log_path, index=False)
    print(f"\n[RL] 训练完成！日志保存至: {log_path}")
    print(f"[RL] 最优 checkpoint: {output_dir / 'rl_best.ckpt'}")
    print(f"[RL] 最优验证 reward: {best_reward:.4f}")


# ─────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        default=str(DEFAULT_HUMAN_EVAL_CSV))
    p.add_argument("--checkpoint", default=str(CHECKPOINTS_DIR / "homo_sapiens" / "finetune.ckpt"))
    p.add_argument("--base_model", default=str(BASE_MODEL_DIR))
    p.add_argument("--organism",   default=DEFAULT_ORGANISM)
    p.add_argument("--output_dir", default=str(CHECKPOINTS_DIR / "homo_sapiens_rl"))
    p.add_argument("--steps",      type=int,   default=3000)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-5)
    p.add_argument("--lambda_mlm", type=float, default=0.1)
    p.add_argument("--w_csi",      type=float, default=1.0)
    p.add_argument("--w_cfd",      type=float, default=1.0)
    p.add_argument("--w_cousin",   type=float, default=1.0)
    p.add_argument("--w_cis",      type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
