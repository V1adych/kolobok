import logging
from dataclasses import dataclass
from typing import Tuple
from copy import deepcopy

import tyro
import torch
from torch import nn
import torchvision
from torchvision.transforms import functional as VF

from rfdetr import RFDETRBase


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("torch2onnx")


@dataclass
class Args:
    ckpt_path: str
    onnx_path: str
    shape: Tuple[int, int] = (560, 560)
    num_classes: int = 3
    num_select: int = 300
    iou_threshold: float = 0.2


class PostProcess(nn.Module):
    def __init__(self, num_select=300):
        super().__init__()
        self.num_select = num_select

    def forward(self, boxes, logits):
        prob = logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(logits.shape[0], -1), self.num_select, dim=1
        )
        topk_boxes = topk_indexes // logits.shape[2]
        labels = topk_indexes % logits.shape[2]

        boxes_selected = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4)
        )

        return boxes_selected, labels, topk_values


def cxcywh2xyxy(boxes):
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


class RFDETR(nn.Module):
    def __init__(self, model: RFDETRBase, num_select=300, iou_threshold=0.2):
        super().__init__()
        model.model.model.cpu()
        model.model.model.eval()
        self.means = model.means
        self.stds = model.stds
        self.model = model.model

        self.model.inference_model = deepcopy(self.model.model)

        self.postprocess = PostProcess(num_select)
        self.iou_threshold = iou_threshold

        self.nms_fn = torch.vmap(torchvision.ops.batched_nms, in_dims=(0, 0, 0, None))

    def forward(self, x: torch.Tensor):
        out = self.model.inference_model(VF.normalize(x, self.means, self.stds))
        boxes, labels, scores = self.postprocess(out["pred_boxes"], out["pred_logits"])

        boxes_xyxy = cxcywh2xyxy(boxes)

        # indices = self.nms_fn(boxes_xyxy, scores, labels, self.iou_threshold)

        # boxes_xyxy = torch.gather(boxes_xyxy, 1, indices.unsqueeze(-1).repeat(1, 1, 4))
        # labels = torch.gather(labels, 1, indices)
        # scores = torch.gather(scores, 1, indices)

        return boxes_xyxy, labels, scores

def disable_grad(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False

@torch.no_grad()
def main():
    args = tyro.cli(Args)

    dummy_input = torch.rand(1, 3, args.shape[0], args.shape[1])

    logger.info(f"Converting {args.ckpt_path} to ONNX...")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model = RFDETRBase()
    model.model_config.device = "cpu"
    model.model.model.to("cpu")
    model.model.device = "cpu"
    model.model.reinitialize_detection_head(num_classes=args.num_classes)
    model.model.model.load_state_dict(ckpt["model"])
    model.export()
    return

    # model = RFDETR(
    #     model, num_select=args.num_select, iou_threshold=args.iou_threshold
    # )
    disable_grad(model.model.inference_model)
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_path,
        verbose=True,
        opset_version=21,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["boxes", "labels", "scores"],
    )

    logger.info(f"Saved {args.onnx_path}")


if __name__ == "__main__":
    main()
