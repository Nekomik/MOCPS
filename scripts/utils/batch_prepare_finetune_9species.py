import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

from mocps.config import BASE_MODEL_DIR, CHECKPOINTS_DIR, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--base_model_dir", type=str, default=str(BASE_MODEL_DIR))
    parser.add_argument("--species", nargs="*", default=None, help="只训练指定 stem，如 arabidopsis_thaliana")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_fraction", type=float, default=0.1)
    parser.add_argument("--save_every_n_steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--force_reprepare_json", action="store_true")
    parser.add_argument("--force_retrain", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    finetune_csv_dir = project_root / "data" / "finetune_csv"
    finetune_json_dir = project_root / "data" / "finetune_json"
    checkpoints_dir = CHECKPOINTS_DIR
    base_model_dir = Path(args.base_model_dir)

    finetune_json_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(
        [
            p for p in finetune_csv_dir.glob("*.csv")
            if not p.name.endswith("_eval.csv") and "summary" not in p.name
        ]
    )
    if args.species:
        wanted = set(args.species)
        csv_files = [p for p in csv_files if p.stem in wanted]

    if not csv_files:
        raise FileNotFoundError(f"no training csv found under: {finetune_csv_dir}")

    summary_rows = []

    for csv_path in csv_files:
        stem = csv_path.stem
        json_path = finetune_json_dir / f"{stem}_training_data.json"
        ckpt_dir = checkpoints_dir / stem
        ckpt_path = ckpt_dir / "finetune.ckpt"
        train_log = ckpt_dir / "train.log"

        print("\n" + "=" * 80)
        print(f"[START] {stem}")
        print("=" * 80)

        if args.force_reprepare_json or (not json_path.exists()):
            from CodonTransformer.CodonData import prepare_training_data

            print(f"[PREPARE] {csv_path} -> {json_path}")
            prepare_training_data(str(csv_path), str(json_path))
        else:
            print(f"[SKIP PREPARE] existing: {json_path}")

        if ckpt_path.exists() and (not args.force_retrain):
            print(f"[SKIP TRAIN] checkpoint exists: {ckpt_path}")
        else:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            with open(train_log, "a") as f:
                f.write(f"\n\n===== {datetime.now().isoformat()} START {stem} =====\n")

            from mocps.training import train_one_species

            train_one_species(
                dataset_json=json_path,
                base_model_dir=base_model_dir,
                checkpoint_dir=ckpt_dir,
                checkpoint_filename="finetune.ckpt",
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                num_workers=args.num_workers,
                accumulate_grad_batches=args.accumulate_grad_batches,
                num_gpus=args.num_gpus,
                learning_rate=args.learning_rate,
                warmup_fraction=args.warmup_fraction,
                save_every_n_steps=args.save_every_n_steps,
                seed=args.seed,
            )

        metrics_csvs = sorted((ckpt_dir / "lightning_logs").glob("**/metrics.csv"))
        summary_rows.append(
            {
                "species": stem,
                "train_csv": str(csv_path),
                "train_json": str(json_path),
                "checkpoint": str(ckpt_path),
                "metrics_csv": str(metrics_csvs[-1]) if metrics_csvs else "",
                "train_log": str(train_log),
            }
        )

    summary_df_path = checkpoints_dir / "batch_train_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_df_path, index=False)
    print(f"\nSaved summary: {summary_df_path}")


if __name__ == "__main__":
    main()