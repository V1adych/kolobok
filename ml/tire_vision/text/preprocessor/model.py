import logging
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import onnxruntime

from torchvision.transforms import functional as VF, InterpolationMode

from tire_vision.config import SidewallSegmentatorConfig


class SegformerWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        logits = self.model(images).logits
        return logits


class SidewallSegmentator:
    def __init__(self, config: SidewallSegmentatorConfig):
        self.config = config
        self.logger = logging.getLogger("sidewall_segmentator")
        self.onnx_path = self.config.segmentator_onnx
        self.session = onnxruntime.InferenceSession(self.onnx_path)

        self.logger.info("SidewallSegmentator initialized successfully")

    @torch.no_grad()
    def forward(self, image: np.ndarray):
        start_time = time.perf_counter()
        torch_image = (
            torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
            / 255
        )
        *_, h, w = torch_image.shape

        np_image = VF.resize(
            torch_image,
            self.config.resize_shape,
            interpolation=InterpolationMode.BICUBIC,
        ).numpy()
        logits = self.session.run(
            None,
            {
                "input": np_image,
            },
        )[0]
        logits = torch.from_numpy(logits)
        logits = VF.resize(
            logits, (h, w), interpolation=InterpolationMode.BICUBIC
        ).squeeze()

        mask = (F.sigmoid(logits) > self.config.confidence_threshold).to(
            torch.uint8
        ).numpy() * 255

        end_time = time.perf_counter()
        self.logger.info(
            f"Completed sidewall segmentation in {end_time - start_time} seconds"
        )

        return mask
