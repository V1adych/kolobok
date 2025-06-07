from pathlib import Path
from typing import Tuple, Optional, Iterable, Callable, List, Dict
from collections import OrderedDict
from copy import deepcopy

import cv2
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import (
    swin_s,
    Swin_S_Weights,
    efficientnet_b7,
    EfficientNet_B7_Weights,
    swin_v2_t,
    Swin_V2_T_Weights,
    swin_v2_s,
    Swin_V2_S_Weights,
    vit_b_16,
    ViT_B_16_Weights,
    densenet201,
    DenseNet201_Weights,
    googlenet,
    GoogLeNet_Weights,
    convnext_small,
    ConvNeXt_Small_Weights,
    regnet_y_8gf,
    RegNet_Y_8GF_Weights,
)
from torchvision.models.swin_transformer import Permute
from torchvision.io import read_image
from torchvision import transforms
from torchvision.ops import FeaturePyramidNetwork

from tqdm import tqdm


data_root = Path("data/dataset_crop")

df = pd.read_csv(data_root / "thread_depths.csv")
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)


class ThreadDataset(Dataset):
    def __init__(self, data: pd.DataFrame, data_root_dir: str, transform: Optional[nn.Module] = None):
        self.data = data
        self.data_root_dir = Path(data_root_dir)
        self.image_paths = []
        self.labels = []
        for _, row in self.data.iterrows():
            image_path = self.data_root_dir / row["path"]
            if not image_path.exists():
                print(f"Warning: {image_path} does not exist")
                continue
            self.image_paths.append(self.data_root_dir / row["path"])
            self.labels.append(row["label"])

        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = read_image(str(image_path))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


image_path = data_root / df.sample().iloc[0, 0]
img = read_image(image_path)


class Clahe(nn.Module):
    def __init__(self, clip_limit: float = 2.0):
        super().__init__()
        self.clip_limit = clip_limit
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit)

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        result = []
        for channel in img:
            result.append(self.clahe.apply(channel))

        return np.stack(result, axis=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x_npy = x.cpu().numpy()
        x_clahe = self._apply_clahe(x_npy)

        return torch.tensor(x_clahe, device=device)


transform = transforms.Compose(
    [
        # Clahe(),
        lambda x: x / 255,
        transforms.Resize(
            (512, 512), interpolation=transforms.InterpolationMode.BICUBIC
        ),
    ]
)

transform_aug = transforms.Compose(
    [
        transform,
        # transforms.ToPILImage(),
        # transforms.RandAugment(
        #     num_ops=3, interpolation=transforms.InterpolationMode.BICUBIC
        # ),
        # transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomAffine(15, (0.05, 0.05), fill=255),
    ]
)

# img = transform_aug(img)

# plt.imshow(img.permute(1, 2, 0))
# plt.axis("off")
# plt.show()


train_dataset = ThreadDataset(df_train, data_root, transform_aug)
val_dataset = ThreadDataset(df_val, data_root, transform)

train_loader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=8)
val_loader = DataLoader(val_dataset, shuffle=False, num_workers=8, batch_size=16)


models = {}

# model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 1))

models["swin_v2"] = swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)
models["swin_v2"].head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))

# model = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
# model.head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))

models["effnet_b7"] = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
models["effnet_b7"].classifier = nn.Sequential(nn.Linear(2560, 512), nn.SiLU(), nn.Linear(512, 1))

# model = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
# model.features[0][0] = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(5, 5), stride=(2, 2))
# model.head = nn.Sequential(nn.Linear(768, 512), nn.GELU(), nn.Linear(512, 1))

# model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
# model.heads = nn.Sequential(nn.Linear(768, 512), nn.GELU(), nn.Linear(512, 1))

models["densenet201"] = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
models["densenet201"].classifier = nn.Sequential(nn.Linear(1920, 512), nn.ReLU(), nn.Linear(512, 1))

models["googlenet"] = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
models["googlenet"].fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))

# model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
# model.classifier[-1] = nn.Sequential(nn.Linear(768, 512), nn.GELU(), nn.Linear(512, 1))

# model = regnet_y_8gf(weights=RegNet_Y_8GF_Weights.IMAGENET1K_V2)
# model.fc = nn.Sequential(nn.Linear(2016, 512), nn.ReLU(), nn.Linear(512, 1))


def train_fn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
):
    best_val_metric = -torch.inf
    best_model = deepcopy(model).cpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=3,
    )

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1} Training",
            total=len(train_loader),
        ):
            images = images.to(device)
            labels = labels.to(device, torch.float32).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(images).exp()
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}"
        )

        model.eval()

        residuals = []
        with torch.no_grad():
            for images, labels in tqdm(
                val_loader, desc=f"Epoch {epoch + 1} Evalutaion"
            ):
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(images).exp()
                residuals.extend(torch.abs(outputs - labels).detach().cpu().tolist())
        residuals = torch.tensor(residuals)
        val_metric = torch.mean(torch.where(residuals <= 1, 1.0, 0.0))
        print(
            f"Validation MAE: {torch.mean(residuals):.4f}, "
            + f"Fraction of errors <= 1: {val_metric:.4f}, "
            + f"0.9th quantile: {torch.quantile(residuals, 0.9):.4f}"
        )
        scheduler.step(val_metric)

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_model = deepcopy(model).cpu()

    model.cpu()
    return best_model


for model_name, model in models.items():
    print(f"TRAINING {model_name}")
    model = train_fn(
        model,
        train_loader,
        val_loader,
        num_epochs=50,
    )
    print()
 
    torch.save(model.state_dict(), f"checkpoints/{model_name}.pt")