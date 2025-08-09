from typing import Dict, Any, Optional, Callable, List
import time
from traceback import format_exc
import json

import numpy as np
import cv2

from tire_vision.text.preprocessor.model import SidewallSegmentator
from tire_vision.text.preprocessor.unwrapper import SidewallUnwrapper
from tire_vision.text.ocr.pipeline import OCRPipeline
from tire_vision.text.index.pipeline import IndexPipeline
from tire_vision.config import TireAnnotationPipelineConfig

import logging


def _json_format(data):
    return json.dumps(data, indent=4)


class TireAnnotationPipeline:
    def __init__(self, config: TireAnnotationPipelineConfig):
        self.config = config
        self.detector = SidewallSegmentator(config.sidewall_segmentator_config)
        self.unwrapper = SidewallUnwrapper(config.sidewall_unwrapper_config)
        self.ocr = OCRPipeline(config.ocr_config)
        self.index = IndexPipeline(config.index_config)

        self.max_image_size = config.max_image_size
        self.image_composition_strategy = config.image_composition_strategy

        self._composition_strategies: Dict[
            str, Callable[[Optional[np.ndarray], np.ndarray], List[np.ndarray]]
        ] = {
            "unwrapped": self._compose_unwrapped,
            "both": self._compose_both,
        }

        self.logger = logging.getLogger("tire_annotation_pipeline")
        self.logger.info("TireAnnotationPipeline initialized")

    def __call__(self, image: np.ndarray) -> Dict[str, Any]:
        self.logger.info("Running TireAnnotationPipeline")
        start_time = time.perf_counter()

        unwrap_success = False
        unwrapped_image: Optional[np.ndarray] = None

        self.logger.info("Running SidewallSegmentator and SidewallUnwrapper")
        try:
            self.logger.info("Running TireDetector")
            tire_mask = self.detector.forward(image)
            self.logger.info(f"TireDetector result shape: {tire_mask.shape}")
            unwrapped_image = self.unwrapper.forward(image, tire_mask)
            unwrap_success = True

        except Exception:
            self.logger.error(format_exc())
            self.logger.error("Error unwrapping tire. Falling back to original image")

        self.logger.info(
            f"Original image shape: {image.shape}, unwrapped image shape: {getattr(unwrapped_image, 'shape', None)}"
        )

        images_for_ocr = self.image_composition(
            unwrapped_image if unwrap_success else None, image
        )

        images_for_ocr = list(map(self._dynamic_resize, images_for_ocr))

        self.logger.info("Running OCRPipeline")
        ocr_result = self.ocr.extract_tire_info(images_for_ocr)
        self.logger.info(f"OCRPipeline result:\n{_json_format(ocr_result)}")

        self.logger.info("Running IndexPipeline")
        index_results = self.index.get_best_matches(ocr_result["strings"])
        self.logger.info(f"IndexPipeline result:\n{_json_format(index_results)}")

        combined_result = {
            "strings": ocr_result["strings"],
            "tire_size": ocr_result["tire_size"],
            "index_results": index_results,
        }

        latency = time.perf_counter() - start_time
        self.logger.info(f"TireAnnotationPipeline completed in {latency:.4f} seconds")

        return combined_result

    def _dynamic_resize(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) > self.max_image_size:
            scale = self.max_image_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return image

    def image_composition(
        self, unwrapped_image: Optional[np.ndarray], original_image: np.ndarray
    ) -> List[np.ndarray]:
        if unwrapped_image is None:
            self.logger.warning(
                "Unwrap failed, using only original image for OCR. This could lead to less accurate results."
            )
            return [original_image]

        compose = self._composition_strategies.get(self.image_composition_strategy)
        if compose is None:
            self.logger.warning(
                f"Unknown image_composition_strategy '{self.image_composition_strategy}', falling back to 'unwrapped'"
            )
            compose = self._compose_unwrapped
        self.logger.info(
            f"Unwrap successful. Using image composition strategy: {compose.__name__}"
        )
        return compose(unwrapped_image, original_image)

    def _compose_unwrapped(
        self, unwrapped_image: Optional[np.ndarray], original_image: np.ndarray
    ) -> List[np.ndarray]:
        return [unwrapped_image]

    def _compose_both(
        self, unwrapped_image: Optional[np.ndarray], original_image: np.ndarray
    ) -> List[np.ndarray]:
        return [unwrapped_image, original_image]
