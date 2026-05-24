from dataclasses import dataclass
import datetime
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import tyro
from torch.utils.data import DataLoader
from train_utils.depth import load_dataset, split_dataset, build_transforms, ThreadDepthDataset, DepthRegressionModule


@dataclass
class Args:
    data_dir: str
    val_fraction: float = 0.1
    model_name: str = "effnet_v2_l"
    as_classification: bool = False
    num_bins: int = 17
    bins_min: float = 1.0
    bins_max: float = 9.0
    size: int = 640
    batch_size: int = 16
    num_epochs: int = 25
    lr: float = 1e-4
    num_workers: int = 8
    seed: int = 42
    ckpt_dir: str = "depth_checkpoints"
    resume_training_checkpoint: Optional[str] = None
    aug: bool = True
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"
    pretrained: bool = True


def main():
    args = tyro.cli(Args)
    pl.seed_everything(args.seed, workers=True)

    image_paths, labels = load_dataset(args.data_dir)
    (train_image_paths, train_labels), (val_image_paths, val_labels) = split_dataset(
        image_paths,
        labels,
        args.val_fraction,
        args.seed,
    )
    train_ds = ThreadDepthDataset(train_image_paths, train_labels, transform=build_transforms(args.size, do_aug=args.aug), depth_range=(args.bins_min, args.bins_max))
    val_ds = ThreadDepthDataset(val_image_paths, val_labels, transform=build_transforms(args.size, do_aug=False), depth_range=(args.bins_min, args.bins_max))

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size:   {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    module = DepthRegressionModule(args)
    ckpt_dir = Path(args.ckpt_dir) / f"checkpoints-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="model-{epoch:03d}-{val_mae:.5f}-{val_frac_le1:.5f}",
        monitor="val_frac_le1",
        mode="max",
        save_top_k=10,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
    )

    print("Starting training...")
    trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume_training_checkpoint)
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
