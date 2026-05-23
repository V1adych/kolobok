from dataclasses import dataclass
import datetime
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import polars as plr
import pytorch_lightning as pl
import torch
import tyro
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    DenseNet201_Weights,
    EfficientNet_B7_Weights,
    EfficientNet_V2_L_Weights,
    GoogLeNet_Weights,
    Swin_S_Weights,
    Swin_V2_T_Weights,
    densenet201,
    efficientnet_b7,
    efficientnet_v2_l,
    googlenet,
    swin_s,
    swin_v2_t,
)


@dataclass
class Args:
    data_dir: str
    val_fraction: float = 0.1
    model_name: str = "effnet_v2_l"
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


def get_model(model_name: str) -> nn.Module:
    if model_name == "swin_s":
        model = swin_s(weights=Swin_S_Weights.DEFAULT)
        model.head = nn.Linear(768, 1)
    elif model_name == "swin_v2_t":
        model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        model.head = nn.Linear(768, 1)
    elif model_name == "effnet_b7":
        model = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(2560, 1)
    elif model_name == "effnet_v2_l":
        model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(1280, 1)
    elif model_name == "densenet201":
        model = densenet201(weights=DenseNet201_Weights.DEFAULT)
        model.classifier = nn.Linear(1920, 1)
    elif model_name == "googlenet":
        model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
        model.fc = nn.Linear(1024, 1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


class ThreadDepthDataset(Dataset):
    def __init__(self, image_paths: list[str], labels: list[float], transform):
        super().__init__()
        self.transform = transform
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise ValueError(f"Failed to load image: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def load_dataset(data_dir: str) -> tuple[list[str], list[float]]:
    root = Path(data_dir)
    labels_path = root / "labels.csv"
    images_dir = root / "images"
    df = plr.read_csv(labels_path)
    pairs = df.select(["image_name", "label"]).to_dicts()
    pairs = [(images_dir / item["image_name"], item["label"]) for item in pairs]
    pairs = [pair for pair in pairs if pair[0].exists()]
    image_paths, labels = map(list, zip(*pairs))
    return image_paths, labels


def split_dataset(
    image_paths: list[str],
    labels: list[float],
    val_fraction: float,
    seed: int,
) -> tuple[tuple[list[str], list[float]], tuple[list[str], list[float]]]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    num_samples = len(image_paths)
    if num_samples == 0:
        raise ValueError("Dataset is empty")
    num_val = max(1, int(round(num_samples * val_fraction)))
    if num_val >= num_samples:
        raise ValueError(f"val_fraction={val_fraction} leaves no training samples for dataset of size {num_samples}")

    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed)).tolist()
    val_indices = set(indices[:num_val])
    train_image_paths = [image_paths[i] for i in range(num_samples) if i not in val_indices]
    train_labels = [labels[i] for i in range(num_samples) if i not in val_indices]
    val_image_paths = [image_paths[i] for i in range(num_samples) if i in val_indices]
    val_labels = [labels[i] for i in range(num_samples) if i in val_indices]
    return (train_image_paths, train_labels), (val_image_paths, val_labels)


def build_transforms(size: int, do_aug: bool):
    transforms = []
    if do_aug:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-20, 20),
                    shear=(-10, 10),
                    fit_output=False,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    p=0.5,
                ),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02, p=0.5),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.MotionBlur(blur_limit=(3, 5), p=1.0),
                        A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
                    ],
                    p=0.25,
                ),
            ]
        )
    transforms.extend(
        [
            A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


class DepthRegressionModule(pl.LightningModule):
    def __init__(self, args: Args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args
        self.model = get_model(args.model_name)
        self.criterion = nn.HuberLoss(delta=1.0)

    @staticmethod
    def _metrics(outputs: torch.Tensor, labels: torch.Tensor):
        residuals = torch.abs(outputs - labels)
        mae = residuals.mean()
        frac_le1 = (residuals <= 1.0).float().mean()
        return mae, frac_le1

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.unsqueeze(1)
        preds = self(images)
        loss = self.criterion(preds, labels)
        mae, frac_le1 = self._metrics(preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_frac_le1", frac_le1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.unsqueeze(1)
        preds = self(images)
        loss = self.criterion(preds, labels)
        mae, frac_le1 = self._metrics(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_frac_le1", frac_le1, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)


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
    train_ds = ThreadDepthDataset(train_image_paths, train_labels, transform=build_transforms(args.size, do_aug=args.aug))
    val_ds = ThreadDepthDataset(val_image_paths, val_labels, transform=build_transforms(args.size, do_aug=False))

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size:   {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    module = DepthRegressionModule(args)
    ckpt_dir = Path(args.ckpt_dir) / f"checkpoints-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="model-{epoch:03d}-{val_mae:.5f}",
        monitor="val_mae",
        mode="min",
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
