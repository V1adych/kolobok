import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import tyro
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from torch import nn

from mmdet.registry import MODELS

DEFAULT_CONFIG_PATH = "submodules/mmdetection/configs/rtmdet/tire_count.py"
ONNX_OPSET_VERSION = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("torch2onnx")


class RTMDetDetectorWrapper(nn.Module):
    def __init__(self, model: nn.Module, input_size: Tuple[int, int]):
        super().__init__()
        self.model = model.eval()
        self.h, self.w = map(int, input_size)
        dp = model.data_preprocessor
        self.register_buffer("mean", torch.as_tensor(dp.mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(dp.std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        x = (x.float() * 255.0 - self.mean) / self.std
        head = self.model.bbox_head
        cls_scores, bbox_preds, kernel_preds, mask_feat = head(self.model.extract_feat(x))
        scores = torch.cat([t.permute(0, 2, 3, 1).reshape(1, -1, head.cls_out_channels) for t in cls_scores], dim=1)
        scores = scores.sigmoid() if head.use_sigmoid_cls else scores.softmax(dim=-1)[..., :-1]
        bbox_preds = torch.cat([t.permute(0, 2, 3, 1).reshape(1, -1, 4) for t in bbox_preds], dim=1)
        kernels = torch.cat([t.permute(0, 2, 3, 1).reshape(1, -1, head.num_gen_params) for t in kernel_preds], dim=1)
        priors = torch.cat(
            head.prior_generator.grid_priors(
                [t.shape[-2:] for t in cls_scores],
                dtype=x.dtype,
                device=x.device,
                with_stride=True,
            ),
            dim=0,
        )[None]
        points = priors[..., :2]
        x1 = (points[..., 0] - bbox_preds[..., 0]).clamp(0, self.w)
        y1 = (points[..., 1] - bbox_preds[..., 1]).clamp(0, self.h)
        x2 = (points[..., 0] + bbox_preds[..., 2]).clamp(0, self.w)
        y2 = (points[..., 1] + bbox_preds[..., 3]).clamp(0, self.h)
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return boxes, scores, kernels, priors, mask_feat


class RTMDetMaskDecoderWrapper(nn.Module):
    def __init__(self, model: nn.Module, max_instances: int):
        super().__init__()
        self.head = model.bbox_head
        self.m = int(max_instances)

    def forward(self, mask_feat: torch.Tensor, kernels: torch.Tensor, priors: torch.Tensor, valid: torch.Tensor):
        logits = self.head._mask_predict_by_feat_single(mask_feat[0], kernels[0], priors[0])
        return (logits * valid[0, :, None, None].to(logits.dtype))[None]


@dataclass
class Args:
    ckpt_path: str
    detector_onnx_path: str
    mask_decoder_onnx_path: str
    input_size: Tuple[int, int] = (640, 640)
    max_mask_instances: int = 16
    config_path: str = DEFAULT_CONFIG_PATH


def load_model(config_path: str, ckpt_path: str) -> nn.Module:
    cfg = Config.fromfile(config_path)
    init_default_scope("mmdet")
    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model.to("cpu")
    model.eval()
    return model


def build_decoder_inputs(
    detector_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    max_mask_instances: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, kernels, priors, mask_feat = detector_outputs
    num_params = kernels.shape[-1]
    kernels_pad = torch.zeros((1, max_mask_instances, num_params), dtype=kernels.dtype)
    priors_pad = torch.zeros((1, max_mask_instances, 4), dtype=priors.dtype)
    valid = torch.zeros((1, max_mask_instances), dtype=kernels.dtype)

    kernels_pad[0] = kernels[0, :max_mask_instances]
    priors_pad[0] = priors[0, :max_mask_instances]
    valid[0] = 1.0
    return mask_feat, kernels_pad, priors_pad, valid


@torch.no_grad()
def main():
    args = tyro.cli(Args)

    detector_onnx_path = Path(args.detector_onnx_path).expanduser()
    mask_decoder_onnx_path = Path(args.mask_decoder_onnx_path).expanduser()
    detector_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    mask_decoder_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading RTMDet model from {args.ckpt_path}...")
    model = load_model(args.config_path, args.ckpt_path)
    det = RTMDetDetectorWrapper(model, args.input_size).eval()
    msk = RTMDetMaskDecoderWrapper(model, args.max_mask_instances).eval()

    x = torch.rand(1, 3, *args.input_size, dtype=torch.float32)
    detector_outputs = det(x)
    decoder_inputs = build_decoder_inputs(detector_outputs, args.max_mask_instances)

    logger.info(f"Exporting detector ONNX to {detector_onnx_path}...")
    torch.onnx.export(
        det,
        x,
        str(detector_onnx_path),
        opset_version=ONNX_OPSET_VERSION,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["boxes", "scores", "kernels", "priors", "mask_feat"],
    )

    logger.info(f"Exporting mask decoder ONNX to {mask_decoder_onnx_path}...")
    torch.onnx.export(
        msk,
        decoder_inputs,
        str(mask_decoder_onnx_path),
        opset_version=ONNX_OPSET_VERSION,
        do_constant_folding=True,
        input_names=["mask_feat", "kernels", "priors", "valid"],
        output_names=["mask_logits"],
    )

    logger.info("RTMDet ONNX export completed successfully")


if __name__ == "__main__":
    main()
