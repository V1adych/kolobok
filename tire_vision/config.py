from dataclasses import dataclass
import os
from typing import Literal, Tuple


@dataclass
class SegmentationConfig:
    device: str = "cuda"
    target: str = "wheel-tire-thread"
    vocab_aug_mode: Literal["COCO-stuff", "COCO-all", "none"] = "COCO-stuff"
    segmentation_mode: Literal["accurate", "efficient"] = "accurate"
    padding_frac: float = 0.01


@dataclass
class SpikePipelineConfig:
    detector_checkpoint: str = os.environ["SPIKE_DETECTOR_CHECKPOINT"]
    classifier_checkpoint: str = os.environ["SPIKE_CLASSIFIER_CHECKPOINT"]
    device: str = "cuda"
    detection_threshold: float = 0.5
    erosion_iterations: int = 3
    dilation_iterations: int = 3
    crop_size: int = 32


@dataclass
class DepthEstimatorConfig:
    model_name: str = os.environ["DEPTH_ESTIMATOR_MODEL_NAME"]
    checkpoint: str = os.environ["DEPTH_ESTIMATOR_CHECKPOINT"]
    device: str = "cuda"
    resize_shape: Tuple[int, int] = (512, 512)


@dataclass
class TireVisionConfig:
    segmentation: SegmentationConfig = SegmentationConfig()
    spikes: SpikePipelineConfig = SpikePipelineConfig()
    depth: DepthEstimatorConfig = DepthEstimatorConfig()
