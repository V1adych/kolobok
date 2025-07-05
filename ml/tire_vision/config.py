from dataclasses import dataclass
import os
from typing import Literal, Tuple, Optional
import warnings

import torch
import cv2

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
    segmentation_mode: Literal["accurate", "efficient"] = "efficient"
    padding_frac: float = 0.01
    min_tire_pixels: int = 96


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


OCR_PROMPT = """Your task is to extract ALL (absolutely all) visible text from the provided image(s) of a tire.
Present the extracted text as a JSON object with a single key "strings", which should be a list of all text strings found on the tire.
Do not include any reasoning or explanations, only the final JSON object.

Example of a valid response:
{
    "strings": ["MICHELIN", "Pilot Sport 4 S", "245/35ZR20", "95Y", "Extra Load"]
}

If no text is visible, return:
{
    "strings": []
}
"""


@dataclass
class TireDetectorConfig:
    model_id: str = os.environ["TIRE_DETECTOR_MODEL_ID"]
    roboflow_api_key: str = os.environ["ROBOFLOW_API_KEY"]


@dataclass
class TireUnwrapperConfig:
    crop_enlarge_factor: float = 1.1
    polar_angle_steps: int = 360
    polar_flags: int = cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS
    perspective_margin: int = 10
    clahe_mode: Literal["disabled", "luminance", "colors", "black-and-white"] = (
        "black-and-white"
    )
    clahe_clip_limit: float = 5.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    cut_strip: bool = True
    cut_mask_threshold: float = 0.5
    concat_strip: bool = True


@dataclass
class OCRConfig:
    model_name: str = "qwen/qwen2.5-vl-72b-instruct:free"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = os.environ["OPENROUTER_API_KEY"]
    prompt: str = OCR_PROMPT
    top_p: float = 0.9
    temperature: float = 0.7
    presence_penalty: float = 0
    frequency_penalty: float = 0
    max_completion_tokens: int = 4096


@dataclass
class TireIndexConfig:
    db_host: str = os.environ.get("DB_HOST", "db")
    db_port: int = int(os.environ.get("DB_PORT", "3306"))
    db_name: str = os.environ["MYSQL_DATABASE"]
    db_user: str = "root"
    db_password: str = os.environ["MYSQL_ROOT_PASSWORD"]
    table_name: str = "models"
    similarity_threshold: float = 0.5
    max_query_results: int = 3


@dataclass
class TireVisionConfig:
    segmentation = SegmentationConfig()
    spikes = SpikePipelineConfig()
    depth = DepthEstimatorConfig()
    tire_detector = TireDetectorConfig()
    tire_unwrapper = TireUnwrapperConfig()
    ocr = OCRConfig()
    tire_index = TireIndexConfig()
