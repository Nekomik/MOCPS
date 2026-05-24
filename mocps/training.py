import json
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BigBirdForMaskedLM

from CodonTransformer.CodonUtils import MAX_LEN, TOKEN2MASK


class JSONLinesDataset(Dataset):
    def __init__(self, json_path: str | Path):
        self.data = []
        with open(json_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MaskedTokenizerCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        tokenized = self.tokenizer(
            [ex["codons"] for ex in examples],
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        seq_len = tokenized["input_ids"].shape[-1]
        species_index = torch.tensor([[ex["organism"]] for ex in examples])
        tokenized["token_type_ids"] = species_index.repeat(1, seq_len)

        inputs = tokenized["input_ids"]
        targets = tokenized["input_ids"].clone()

        prob_matrix = torch.full(inputs.shape, 0.15)
        prob_matrix[torch.where(inputs < 5)] = 0.0
        selected = torch.bernoulli(prob_matrix).bool()

        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        inputs[replaced] = torch.tensor(
            list(map(TOKEN2MASK.__getitem__, inputs[replaced].cpu().numpy()))
        )

        randomized = (
            torch.bernoulli(torch.full(selected.shape, 0.1)).bool() & selected & ~replaced
        )
        random_idx = torch.randint(26, 90, prob_matrix.shape, dtype=torch.long)
        inputs[randomized] = random_idx[randomized]

        tokenized["input_ids"] = inputs
        tokenized["labels"] = torch.where(selected, targets, -100)
        return tokenized


class TrainHarness(pl.LightningModule):
    def __init__(self, model, learning_rate: float, warmup_fraction: float):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_fraction = warmup_fraction

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.warmup_fraction,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        self.model.bert.set_attention_type("block_sparse")
        outputs = self.model(**batch)
        batch_size = batch["input_ids"].shape[0]
        self.log(
            "train_loss",
            outputs.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )
        return outputs.loss


class DumpStateDict(pl.callbacks.ModelCheckpoint):
    def __init__(self, checkpoint_dir, checkpoint_filename, every_n_train_steps):
        super().__init__(dirpath=checkpoint_dir, every_n_train_steps=every_n_train_steps)
        self.checkpoint_filename = checkpoint_filename

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        model = trainer.model.model
        torch.save(model.state_dict(), os.path.join(self.dirpath, self.checkpoint_filename))


def train_one_species(
    dataset_json: str | Path,
    base_model_dir: str | Path,
    checkpoint_dir: str | Path,
    checkpoint_filename: str = "finetune.ckpt",
    batch_size: int = 6,
    max_epochs: int = 15,
    num_workers: int = 5,
    accumulate_grad_batches: int = 1,
    num_gpus: int = 1,
    learning_rate: float = 5e-5,
    warmup_fraction: float = 0.1,
    save_every_n_steps: int = 512,
    seed: int = 123,
    debug: bool = False,
    logger=True,
    initial_checkpoint: str | Path | None = None,
):
    pl.seed_everything(seed)
    torch.set_float32_matmul_precision("medium")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), local_files_only=True)
    model = BigBirdForMaskedLM.from_pretrained(str(base_model_dir), local_files_only=True)
    if initial_checkpoint is not None:
        state_dict = torch.load(initial_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
    harnessed_model = TrainHarness(model, learning_rate, warmup_fraction)

    train_data = JSONLinesDataset(dataset_json)
    effective_workers = 0 if debug else num_workers
    data_loader = DataLoader(
        dataset=train_data,
        collate_fn=MaskedTokenizerCollator(tokenizer),
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        persistent_workers=effective_workers > 0,
    )

    save_checkpoint = DumpStateDict(
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_filename=checkpoint_filename,
        every_n_train_steps=save_every_n_steps,
    )

    use_single_device = num_gpus == 1
    trainer = pl.Trainer(
        default_root_dir=str(checkpoint_dir),
        logger=logger,
        strategy="auto" if use_single_device else "ddp_find_unused_parameters_true",
        accelerator="gpu",
        devices=1 if use_single_device else num_gpus,
        precision="16-mixed",
        max_epochs=max_epochs,
        deterministic=False,
        enable_checkpointing=True,
        callbacks=[save_checkpoint],
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=1,
    )
    trainer.fit(harnessed_model, data_loader)
