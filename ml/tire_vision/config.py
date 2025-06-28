from dataclasses import dataclass
import os
from typing import Literal, Tuple
import warnings

import torch

DEVICE = os.environ["DEVICE"]
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    warnings.warn("CUDA is not available, using CPU")
    DEVICE = "cpu"


CLASS_MAPPING = {
    0: "good",
    1: "bad",
}

CLASS_COLORS = {
    "good": (0, 255, 0),
    "bad": (255, 0, 0),
}

@dataclass
class SegmentationConfig:
    device: str = DEVICE
    target: str = "wheel-tire-thread"
    vocab_aug_mode: Literal["COCO-stuff", "COCO-all", "none"] = "COCO-stuff"
    segmentation_mode: Literal["accurate", "efficient"] = "accurate"
    padding_frac: float = 0.01


@dataclass
class SpikePipelineConfig:
    detector_checkpoint: str = os.environ["SPIKE_DETECTOR_CHECKPOINT"]
    classifier_checkpoint: str = os.environ["SPIKE_CLASSIFIER_CHECKPOINT"]
    device: str = DEVICE
    detection_threshold: float = 0.5
    erosion_iterations: int = 3
    dilation_iterations: int = 3
    crop_size: int = 32


@dataclass
class DepthEstimatorConfig:
    model_name: str = os.environ["DEPTH_ESTIMATOR_MODEL_NAME"]
    checkpoint: str = os.environ["DEPTH_ESTIMATOR_CHECKPOINT"]
    device: str = DEVICE
    resize_shape: Tuple[int, int] = (512, 512)


@dataclass
class TireVisionConfig:
    segmentation: SegmentationConfig = SegmentationConfig()
    spikes: SpikePipelineConfig = SpikePipelineConfig()
    depth: DepthEstimatorConfig = DepthEstimatorConfig()
