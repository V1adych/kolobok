import logging
from dataclasses import dataclass
from typing import Tuple

import tyro
import torch
from torch import nn
from torchvision.models import (
    densenet201,
    googlenet,
    swin_v2_t,
    swin_s,
    efficientnet_b7,
    efficientnet_v2_l,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("torch2onnx")


def get_regression_model(model_name: str) -> nn.Module:
    if model_name == "swin_s":
        model = swin_s()
        model.head = nn.Linear(768, 1)
    elif model_name == "swin_v2_t":
        model = swin_v2_t()
        model.head = nn.Linear(768, 1)
    elif model_name == "effnet_b7":
        model = efficientnet_b7()
        model.classifier[1] = nn.Linear(2560, 1)
    elif model_name == "effnet_v2_l":
        model = efficientnet_v2_l()
        model.classifier[1] = nn.Linear(1280, 1)
    elif model_name == "densenet201":
        model = densenet201()
        model.classifier = nn.Linear(1920, 1)
    elif model_name == "googlenet":
        model = googlenet(aux_logits=False)
        model.fc = nn.Linear(1024, 1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def get_classification_model(model_name: str, num_bins: int) -> nn.Module:
    if model_name == "swin_s":
        model = swin_s()
        model.head = nn.Linear(768, num_bins)
    elif model_name == "swin_v2_t":
        model = swin_v2_t()
        model.head = nn.Linear(768, num_bins)
    elif model_name == "effnet_b7":
        model = efficientnet_b7()
        model.classifier[1] = nn.Linear(2560, num_bins)
    elif model_name == "effnet_v2_l":
        model = efficientnet_v2_l()
        model.classifier[1] = nn.Linear(1280, num_bins)
    elif model_name == "densenet201":
        model = densenet201()
        model.classifier = nn.Linear(1920, num_bins)
    elif model_name == "googlenet":
        model = googlenet(aux_logits=False)
        model.fc = nn.Linear(1024, num_bins)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def get_model(model_name: str, is_classification: bool = False, num_bins: int = 11) -> nn.Module:
    if is_classification:
        return get_classification_model(model_name, num_bins)
    else:
        return get_regression_model(model_name)


def load_checkpoint(ckpt_path: str, pl_prefix: str = "model."):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" in state_dict:
        state_dict = {k[len(pl_prefix) :] if k.startswith(pl_prefix) else k: v for k, v in state_dict["state_dict"].items()}

    return state_dict


def get_depth_estimator(model_name: str, checkpoint_path: str, is_classification: bool = False, num_bins: int = 11):
    model = get_model(model_name, is_classification=is_classification, num_bins=num_bins)
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, is_classification: bool = False, num_bins: int = 11, min_bin: float = 1.0, max_bin: float = 9.0):
        super().__init__()
        self.model = model
        self.is_classification = is_classification
        self.num_bins = num_bins
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        if is_classification:
            self.register_buffer("bins", torch.linspace(min_bin, max_bin, num_bins, dtype=torch.float32))
        else:
            self.bins = None

    def forward(self, x: torch.Tensor):
        x = (x - self.input_mean) / self.input_std
        y = self.model(x)
        if self.is_classification:
            probs = torch.softmax(y, dim=1)
            return torch.sum(probs * self.bins.unsqueeze(0), dim=1, keepdim=True)
        else:
            return y.clip(self.min_bin, self.max_bin)


@dataclass
class Args:
    ckpt_path: str
    onnx_path: str
    model_name: str = "googlenet"
    input_shape: Tuple[int, int] = (320, 320)
    is_classification: bool = False
    num_bins: int = 11
    min_bin: float = 1.0
    max_bin: float = 9.0


@torch.no_grad()
def main():
    args = tyro.cli(Args)

    dummy_input = torch.rand(1, 3, *args.input_shape)

    logger.info(f"Converting {args.ckpt_path} to ONNX...")
    model = get_depth_estimator(
        args.model_name,
        args.ckpt_path,
        is_classification=args.is_classification,
        num_bins=args.num_bins,
    )
    model = ModelWrapper(
        model,
        is_classification=args.is_classification,
        num_bins=args.num_bins,
        min_bin=args.min_bin,
        max_bin=args.max_bin,
    )

    model.eval()
    batch_dim = torch.export.Dim("batch", min=1, max=64)

    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_path,
        verbose=True,
        opset_version=21,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
        dynamic_shapes={"x": {0: batch_dim}},
    )

    logger.info(f"Saved {args.onnx_path}")


if __name__ == "__main__":
    main()
