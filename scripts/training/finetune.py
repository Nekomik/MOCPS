"""
Finetune the CodonTransformer model.
"""

import argparse

from mocps.config import BASE_MODEL_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune the CodonTransformer model.")
    parser.add_argument("--dataset_dir", "--dataset_json", dest="dataset_dir", type=str, required=True, help="Dataset JSONL path")
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="Directory where checkpoints are saved"
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default=str(BASE_MODEL_DIR),
        help="Base CodonTransformer model directory",
    )
    parser.add_argument(
        "--checkpoint_filename",
        type=str,
        default="finetune.ckpt",
        help="Filename for the saved checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=15, help="Maximum number of epochs")
    parser.add_argument("--num_workers", type=int, default=5, help="DataLoader workers")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_fraction", type=float, default=0.1)
    parser.add_argument("--save_every_n_steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def main(args):
    from mocps.training import train_one_species

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
        logger=False,
    )


if __name__ == "__main__":
    main(parse_args())
