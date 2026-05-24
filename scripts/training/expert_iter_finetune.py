"""
expert_iter_finetune.py
在 CSI Fine-tuned 模型基础上，用 ICR 优化序列继续训练（Expert Iteration）。
"""

import argparse
from pathlib import Path

from mocps.config import BASE_MODEL_DIR, CHECKPOINTS_DIR, DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Expert Iteration model from CSI fine-tuned checkpoint")
    parser.add_argument("--base_model_dir", default=str(BASE_MODEL_DIR), help="Base CodonTransformer model directory")
    parser.add_argument("--pretrain_checkpoint", default=str(CHECKPOINTS_DIR / "homo_sapiens" / "finetune.ckpt"))
    parser.add_argument("--dataset_dir", "--dataset_json", dest="dataset_dir", default=str(DATA_DIR / "finetune_json" / "expert_iter_homo_sapiens.jsonl"))
    parser.add_argument("--checkpoint_dir", default=str(CHECKPOINTS_DIR / "homo_sapiens_expert_iter"))
    parser.add_argument("--checkpoint_filename", default="expert_iter.ckpt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_fraction", type=float, default=0.1)
    parser.add_argument("--save_every_n_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def main():
    args = parse_args()

    from mocps.training import train_one_species

    print(f"加载CSI Fine-tuned权重: {args.pretrain_checkpoint}")

    train_one_species(
        dataset_json=args.dataset_dir,
        base_model_dir=args.base_model_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_filename=args.checkpoint_filename,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_gpus=args.num_gpus,
        learning_rate=args.learning_rate,
        warmup_fraction=args.warmup_fraction,
        save_every_n_steps=args.save_every_n_steps,
        seed=args.seed,
        debug=args.debug,
        logger=True,
        initial_checkpoint=args.pretrain_checkpoint,
    )

    print("\n训练完成")
    print(f"  最终权重: {Path(args.checkpoint_dir) / args.checkpoint_filename}")
    print(f"  运行: python run_benchmark_expert.py --expert_checkpoint {Path(args.checkpoint_dir) / args.checkpoint_filename}")


if __name__ == "__main__":
    main()
