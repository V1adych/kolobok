import logging
from dataclasses import dataclass
from typing import Tuple

import tyro
import torch
from torch import nn

from train_utils.depth import DepthRegressionModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("torch2onnx")


@dataclass
class Args:
    ckpt_path: str
    onnx_path: str
    model_name: str = "googlenet"
    input_shape: Tuple[int, int] = (320, 320)
    as_classification: bool = False
    num_bins: int = 11
    bins_min: float = 1.0
    bins_max: float = 9.0
    pretrained: bool = False


def get_depth_estimator(cfg: Args):
    model = DepthRegressionModule(cfg)

    model.load_state_dict(torch.load(cfg.ckpt_path, map_location="cpu")["state_dict"])
    model.eval()
    return model


class ModelWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        x = (x - self.input_mean) / self.input_std
        return self.model.predict(x)


@torch.no_grad()
def main():
    args = tyro.cli(Args)

    dummy_input = torch.rand(1, 3, *args.input_shape)

    logger.info(f"Converting {args.ckpt_path} to ONNX...")
    model = get_depth_estimator(args)
    model = ModelWrapper(model)

    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_path,
        verbose=True,
        opset_version=21,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    logger.info(f"Saved {args.onnx_path}")


if __name__ == "__main__":
    main()
