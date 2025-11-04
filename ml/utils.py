from typing import Any, Dict, List, Optional

import numpy as np
import cv2

from tire_vision.thread.pipeline import TireThreadPipeline
from tire_vision.text.pipeline import TireAnnotationPipeline
from tire_vision.config import TireVisionConfig, CLASS_COLORS, CLASS_MAPPING
from tire_vision.options import TireThreadPipelineOptions, TireAnnotationPipelineOptions

from logs_manager import log_wrapper


cfg = TireVisionConfig()
thread_pipeline = TireThreadPipeline(cfg.thread_pipeline_config)
annotation_pipeline = TireAnnotationPipeline(cfg.annotation_pipeline_config)


@log_wrapper
def get_thread_stats(
    image: np.ndarray, options: Optional[TireThreadPipelineOptions] = None
) -> Dict[str, Any]:
    result = thread_pipeline(image, options=options)
    for i in range(len(result["studs"])):
        result["studs"][i]["label"] = CLASS_MAPPING[result["studs"][i]["class"]]

    return result


@log_wrapper
def extract_tire_info(
    image: np.ndarray, options: Optional[TireAnnotationPipelineOptions] = None
) -> Dict[str, Any]:
    result = annotation_pipeline(image, options=options)

    return result


def add_annotations(image: np.ndarray, annotations: List[Dict[str, Any]]) -> np.ndarray:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for annotation in annotations:
        x, y, w, h = annotation["box"]
        color = CLASS_COLORS[annotation["class"]]
        cv2.rectangle(
            image_bgr, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb
