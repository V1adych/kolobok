import logging
from dataclasses import dataclass
from typing import Tuple

import tyro
import shutil
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("torch2onnx")




@dataclass
class Args:
    ckpt_path: str
    onnx_path: str
    input_shape: Tuple[int, int] = (640, 640)


@torch.no_grad()
def main():
    args = tyro.cli(Args)

    logger.info(f"Converting {args.ckpt_path} to ONNX...")
    model = YOLO(args.ckpt_path)
    path = model.export(format="onnx", simplify=True, opset=21, imgsz=args.input_shape, dynamic=False, batch=1)

    shutil.move(path, args.onnx_path)
    logger.info(f"Saved {args.onnx_path}")


if __name__ == "__main__":
    main()
