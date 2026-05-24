from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import tyro
from torch.utils.data import DataLoader
from train_utils.depth import load_dataset, split_dataset, build_transforms, ThreadDepthDataset, DepthRegressionModule


@dataclass
class Args:
    data_dir: str
    ckpt_path: str
    val_fraction: float = 0.1
    model_name: str = "effnet_v2_l"
    as_classification: bool = False
    num_bins: int = 17
    bins_min: float = 1.0
    bins_max: float = 9.0
    size: int = 640
    batch_size: int = 4
    seed: int = 42
    pretrained: bool = False


def main():
    args = tyro.cli(Args)
    pl.seed_everything(args.seed, workers=True)

    image_paths, labels = load_dataset(args.data_dir)
    _, (val_image_paths, val_labels) = split_dataset(
        image_paths,
        labels,
        args.val_fraction,
        args.seed,
    )
    val_ds = ThreadDepthDataset(val_image_paths, val_labels, transform=build_transforms(args.size, do_aug=False), depth_range=(args.bins_min, args.bins_max))

    print(f"Val dataset size:   {len(val_ds)}")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    module = DepthRegressionModule(args)
    module.load_state_dict(torch.load(args.ckpt_path, map_location="cpu")["state_dict"])
    module.eval()

    trainer = pl.Trainer(
        max_epochs=10,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print("Starting evaluation...")
    trainer.test(module, val_loader)


if __name__ == "__main__":
    main()
