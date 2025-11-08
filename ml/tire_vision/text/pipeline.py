from typing import Dict, Any, Optional, Callable, List, Tuple
import time
import re

import numpy as np
import cv2

from tire_vision.text.preprocessor.model import SidewallSegmentator
from tire_vision.text.preprocessor.unwrapper import SidewallUnwrapper
from tire_vision.text.ocr.pipeline import OCRPipeline, OCRResult
from tire_vision.text.index.pipeline import IndexPipeline
from tire_vision.config import TireAnnotationPipelineConfig
from tire_vision.options import TireAnnotationPipelineOptions
from models import AnnotationResult

import logging


class TireAnnotationPipeline:
    def __init__(self, config: TireAnnotationPipelineConfig):
        self.config = config
        self.segmentator = SidewallSegmentator(config.sidewall_segmentator_config)
        self.unwrapper = SidewallUnwrapper(config.sidewall_unwrapper_config)
        self.ocr = OCRPipeline(config.ocr_config)
        self.index = IndexPipeline(config.index_config)

        self.max_image_size = config.max_image_size

        self._composition_strategies: Dict[str, Callable[[Optional[np.ndarray], np.ndarray], List[np.ndarray]]] = {
            "unwrapped": self._compose_unwrapped,
            "both": self._compose_both,
        }

        self.logger = logging.getLogger("tire_annotation_pipeline")
        self.logger.info("TireAnnotationPipeline initialized")

    def __call__(self, image: np.ndarray, options: Optional[TireAnnotationPipelineOptions] = None) -> Dict[str, Any]:
        self.logger.info("Running TireAnnotationPipeline")
        segmentator_options = options.sidewall_segmentator_options if options is not None else None
        unwrapper_options = options.sidewall_unwrapper_options if options is not None else None
        ocr_options = options.ocr_options if options is not None else None
        index_options = options.index_options if options is not None else None
        start_time = time.perf_counter()

        self.logger.info("Running SidewallSegmentator")
        tire_mask = self.segmentator(image, options=segmentator_options)
        self.logger.info(f"SidewallSegmentator result shape: {tire_mask.shape}")

        self.logger.info("Running SidewallUnwrapper")
        unwrapped_image = self.unwrapper(image, tire_mask, options=unwrapper_options)
        self.logger.info(f"original image shape: {image.shape}, unwrapped image shape: {unwrapped_image.shape}")

        images_for_ocr, prompt = self.image_composition(unwrapped_image, image)
        images_for_ocr = list(map(self._dynamic_resize, images_for_ocr))

        self.logger.info("Running OCRPipeline")
        ocr_result = self.ocr(images_for_ocr, prompt, options=ocr_options)
        ocr_result = self._postprocess_ocr_result(ocr_result)
        self.logger.info("OCRPipeline result:")
        self.logger.info(f"strings: {ocr_result.strings}")
        self.logger.info(f"tire_size: {ocr_result.tire_size}")

        self.logger.info("Running IndexPipeline")
        index_results = self.index(ocr_result.strings, options=index_options)
        self.logger.info("IndexPipeline results:")
        results_for_logging = list(
            map(lambda r: (r.brand_name, r.model_name, round(r.combined_score, 2)), index_results)
        )
        self.logger.info(f"(brand, model, score): {results_for_logging}")

        combined_result = AnnotationResult(
            strings=ocr_result.strings, tire_size=ocr_result.tire_size, index_results=index_results
        )

        latency = time.perf_counter() - start_time
        self.logger.info(f"TireAnnotationPipeline completed in {latency:.4f} seconds")

        return combined_result

    def _get_tire_size_matches(self, strings: List[str]) -> List[str]:
        matches = [
            (string, match.group(0))
            for pattern in self.config.tire_size_regex
            for string in strings
            for match in re.finditer(pattern, string)
        ]
        return list(dict.fromkeys(matches))

    def _postprocess_ocr_result(self, ocr_result: OCRResult) -> OCRResult:
        matches = self._get_tire_size_matches(ocr_result.strings)
        if matches:
            matched_strings, matched_patterns = zip(*matches)
            ocr_result.strings = list(set(ocr_result.strings) - set(matched_strings))

            if not ocr_result.tire_size:
                ocr_result.tire_size = matched_patterns[0]

        return ocr_result

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
    ) -> Tuple[List[np.ndarray], str]:
        if unwrapped_image is None:
            self.logger.warning(
                "Unwrap failed, using only original image for OCR. This could lead to less accurate results."
            )
            return [original_image], self.config.original_prompt

        compose = self._composition_strategies[self.config.image_composition_strategy]
        self.logger.info(f"Unwrap successful. Using image composition strategy: {compose.__name__}")
        return compose(unwrapped_image, original_image)

    def _compose_unwrapped(
        self, unwrapped_image: Optional[np.ndarray], original_image: np.ndarray
    ) -> Tuple[List[np.ndarray], str]:
        return [unwrapped_image], self.config.unwrap_prompt

    def _compose_both(
        self, unwrapped_image: Optional[np.ndarray], original_image: np.ndarray
    ) -> Tuple[List[np.ndarray], str]:
        return [unwrapped_image, original_image], self.config.both_prompt
