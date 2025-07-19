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
    efficientnet_b3,
    efficientnet_b7,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("torch2onnx")


def get_swin_v2():
    model = swin_v2_t()
    model.head = nn.Sequential(nn.Linear(768, 512), nn.GELU(), nn.Linear(512, 1))
    return model


def get_swin_s():
    model = swin_s()
    model.head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))
    return model


def get_effnet_b3():
    model = efficientnet_b3()
    model.classifier = nn.Sequential(nn.Linear(1536, 512), nn.SiLU(), nn.Linear(512, 1))
    return model


def get_effnet_b7():
    model = efficientnet_b7()
    model.classifier = nn.Sequential(nn.Linear(2560, 512), nn.SiLU(), nn.Linear(512, 1))
    return model


def get_densenet201():
    model = densenet201()
    model.classifier = nn.Sequential(nn.Linear(1920, 512), nn.ReLU(), nn.Linear(512, 1))
    return model


def get_googlenet():
    model = googlenet()
    model.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))
    return model


models = {
    "swin_v2": get_swin_v2,
    "swin_s": get_swin_s,
    "effnet_b3": get_effnet_b3,
    "effnet_b7": get_effnet_b7,
    "densenet201": get_densenet201,
    "googlenet": get_googlenet,
}


def get_depth_estimator(model_name: str, checkpoint_path: str):
    model = models[model_name]()
    if checkpoint_path:
        model.load_state_dict(
            torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        )
    else:
        logger.warning("Depth estimator checkpoint not found, using random weights")
    model.eval()
    return model


@dataclass
class ModelConfig:
    model_name: str
    ckpt_path: str
    onnx_path: str
    input_shape: Tuple[int, int] = (512, 512)


@dataclass
class Args:
    depth: ModelConfig


@torch.no_grad()
def main():
    args = tyro.cli(Args)

    dummy_input = torch.rand(1, 3, *args.depth.input_shape)

    logger.info(f"Converting {args.depth.ckpt_path} to ONNX...")
    model = get_depth_estimator(args.depth.model_name, args.depth.ckpt_path)

    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        args.depth.onnx_path,
        verbose=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(f"Saved {args.depth.onnx_path}")


if __name__ == "__main__":
    main()
