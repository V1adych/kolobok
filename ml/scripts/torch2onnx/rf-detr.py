import logging
from dataclasses import dataclass
from typing import Tuple

import tyro
import torch

from rfdetr import RFDETRBase


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("torch2onnx")


@dataclass
class Args:
    ckpt_path: str
    shape: Tuple[int, int] = (560, 560)
    num_classes: int = 3
    num_select: int = 300
    iou_threshold: float = 0.2


@torch.no_grad()
def main():
    args = tyro.cli(Args)

    logger.info(f"Converting {args.ckpt_path} to ONNX...")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model = RFDETRBase()
    model.model_config.device = "cpu"
    model.model.model.to("cpu")
    model.model.device = "cpu"
    model.model.reinitialize_detection_head(num_classes=args.num_classes)
    model.model.model.load_state_dict(ckpt["model"])
    model.model.model.eval()
    model.export(opset_version=21)


if __name__ == "__main__":
    main()
