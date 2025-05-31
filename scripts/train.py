from pathlib import Path
from typing import Tuple, Optional
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_s, Swin_S_Weights
from torchvision.io import read_image
from torchvision import transforms

from tqdm import tqdm

from inference import get_model


class ThreadDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        data_root_dir: str,
        transform: Optional[nn.Module] = None,
    ):
        self.data = data
        self.data_root_dir = Path(data_root_dir)
        self.image_paths = []
        self.labels = []
        for _, row in self.data.iterrows():
            image_path = self.data_root_dir / row["image"]
            if not image_path.exists():
                print(f"Warning: {image_path} does not exist")
                continue
            self.image_paths.append(self.data_root_dir / row["image"])
            self.labels.append(row["label"])

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = read_image(str(image_path)).to(torch.float32) / 255
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def train_fn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    grad_accumulation_steps: int = 1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch + 1} Training",
            total=len(train_loader),
        ):
            images = images.to(device)
            labels = labels.to(device, torch.float32).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images).exp()
            loss = criterion(outputs, labels) / grad_accumulation_steps
            loss.backward()

            if (i + 1) % grad_accumulation_steps == 0:
                optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}"
        )

        model.eval()
        mae = []
        within_tolerance = []
        with torch.no_grad():
            for images, labels in tqdm(
                val_loader, desc=f"Epoch {epoch + 1} Evalutaion"
            ):
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(images).exp()
                mae.append(torch.abs(outputs - labels).mean().item())
                within_tolerance.append(
                    int(torch.abs(outputs - labels).squeeze().item() <= 1)
                )
        print(
            f"Validation MAE: {np.mean(mae):.4f}, Fraction of errors <= 1: {np.mean(within_tolerance)}"
        )


def main():
    data_root = Path("data/dataset")
    processed_data_root = Path("data/processed_dataset")

    df = pd.read_csv(processed_data_root / "labels.csv")

    dataset = ThreadDataset(df, processed_data_root)

    augmentation_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandAugment(
                num_ops=4, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [0.8, 0.2],
        generator=torch.Generator().manual_seed(2),
    )
    train_dataset.dataset.transform = augmentation_transform

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=8)

    model = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
    model.head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))

    print("TRAINING FOR 25 EPOCHS")
    train_fn(
        model,
        train_loader,
        val_loader,
        num_epochs=25,
        grad_accumulation_steps=1,
    )


if __name__ == "__main__":
    main()
