from typing import Dict, Any
import time
from traceback import format_exc

import numpy as np


from tire_vision.text.preprocessor.model import SidewallSegmentator
from tire_vision.text.preprocessor.unwrapper import SidewallUnwrapper
from tire_vision.text.ocr.pipeline import OCRPipeline
from tire_vision.text.index.pipeline import IndexPipeline
from tire_vision.config import (
    OCRConfig,
    SidewallUnwrapperConfig,
    SidewallSegmentatorConfig,
    IndexConfig,
)

import logging


class TireAnnotationPipeline:
    def __init__(
        self,
        detector_config: SidewallSegmentatorConfig,
        unwrapper_config: SidewallUnwrapperConfig,
        ocr_config: OCRConfig,
        index_config: IndexConfig,
    ):
        self.detector = SidewallSegmentator(detector_config)
        self.unwrapper = SidewallUnwrapper(unwrapper_config)
        self.ocr = OCRPipeline(ocr_config)
        self.index = IndexPipeline(index_config)

        self.logger = logging.getLogger("tire_annotation_pipeline")
        self.logger.info("TireAnnotationPipeline initialized")

    def __call__(self, image: np.ndarray) -> Dict[str, Any]:
        self.logger.info("Running TireAnnotationPipeline")
        start_time = time.perf_counter()

        unwrap_success = False
        unwrapped_image: np.ndarray | None = None

        self.logger.info("Running TireUnwrapper")
        try:
            self.logger.info("Running TireDetector")
            tire_mask = self.detector.detect(image)
            self.logger.info(f"TireDetector result shape: {tire_mask.shape}")
            unwrapped_image = self.unwrapper.get_unwrapped_tire(image, tire_mask)
            unwrap_success = True
        except Exception:
            self.logger.error(format_exc())
            self.logger.error(
                "Error running TireUnwrapper. Falling back to original image"
            )

        self.logger.info(
            f"Original image shape: {image.shape}, unwrapped image shape: {getattr(unwrapped_image, 'shape', None)}"
        )

        if unwrap_success and unwrapped_image is not None:
            self.logger.info("Unwrap successful, using both images for OCR")
            images_for_ocr = [image, unwrapped_image]
        else:
            self.logger.info("Unwrap failed, using only original image for OCR")
            images_for_ocr = [image]

        self.logger.info("Running TireOCR")
        ocr_result = self.ocr.extract_tire_info(images_for_ocr)
        self.logger.info(f"TireOCR result:\n {ocr_result}")

        self.logger.info("Running TireIndexPipeline")
        index_result = self.index.run(queries=ocr_result["strings"])
        self.logger.info(f"TireIndexPipeline result:\n {index_result}")

        latency = time.perf_counter() - start_time
        self.logger.info(f"TireAnnotationPipeline completed in {latency:.4f} seconds")

        return index_result
