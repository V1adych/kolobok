from typing import Any, Dict, List

import numpy as np
import cv2

from tire_vision.thread.pipeline import TireThreadPipeline
from tire_vision.text.pipeline import TireAnnotationPipeline
from tire_vision.config import (
    TireVisionConfig,
    CLASS_MAPPING,
    CLASS_COLORS,
)

from logs_manager import log_wrapper


cfg = TireVisionConfig()
thread_pipeline = TireThreadPipeline(cfg.thread_pipeline_config)
annotation_pipeline = TireAnnotationPipeline(cfg.annotation_pipeline_config)


@log_wrapper
def get_thread_stats(image: np.ndarray) -> Dict[str, Any]:
    result = thread_pipeline(image)
    return result


@log_wrapper
def extract_tire_info(image: np.ndarray) -> Dict[str, Any]:
    """Extract tire information using OCR."""
    return annotation_pipeline(image)


def add_annotations(image: np.ndarray, annotations: List[Dict[str, Any]]) -> np.ndarray:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for annotation in annotations:
        x, y, w, h = annotation["box"]
        color = CLASS_COLORS[annotation["class"]]
        cv2.rectangle(image_bgr, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb
