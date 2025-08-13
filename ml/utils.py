from typing import Any

import numpy as np
import cv2

# from tire_vision.thread.pipeline import TireThreadPipeline
from tire_vision.text.pipeline import TireAnnotationPipeline
from tire_vision.config import (
    TireVisionConfig,
    CLASS_MAPPING,
    CLASS_COLORS,
    TireAnnotationPipelineConfig,
    TireThreadPipelineConfig,
)


cfg = TireVisionConfig()
# thread_pipeline = TireThreadPipeline(cfg.thread_pipeline_config)
annotation_pipeline = TireAnnotationPipeline(cfg.annotation_pipeline_config)


def get_thread_stats(image: np.ndarray) -> dict[str, Any]:
    # result = thread_pipeline(image)
    # return result
    return {}


def extract_tire_info(image: np.ndarray) -> dict[str, Any]:
    """Extract tire information using OCR."""
    return annotation_pipeline(image)


def add_annotations(image: np.ndarray, annotations: list[dict[str, Any]]) -> np.ndarray:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for annotation in annotations:
        x, y, w, h = annotation["box"]
        class_name = CLASS_MAPPING[annotation["class"]]
        color = CLASS_COLORS[class_name]
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            image_bgr, class_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb
