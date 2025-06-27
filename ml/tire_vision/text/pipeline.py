from typing import Dict, Any

import numpy as np

from tire_vision.text.preprocessor.model import TireDetector
from tire_vision.text.preprocessor.unwrapper import TireUnwrapper
from tire_vision.text.ocr.pipeline import TireOCR
from tire_vision.config import (
    OCRConfig,
    TireUnwrapperConfig,
    TireDetectorConfig,
)


class TireAnnotationPipeline:
    def __init__(
        self,
        detector_config: TireDetectorConfig,
        unwrapper_config: TireUnwrapperConfig,
        ocr_config: OCRConfig,
    ):
        self.detector = TireDetector(detector_config)
        self.unwrapper = TireUnwrapper(unwrapper_config)
        self.ocr = TireOCR(ocr_config)

    def __call__(self, image: np.ndarray) -> Dict[str, Any]:
        detection_result = self.detector.detect(image)
        unwrapped_image = self.unwrapper.get_unwrapped_tire(
            image,
            detection_result[self.detector.tire_class_name],
            detection_result[self.detector.rim_class_name],
        )
        ocr_result = self.ocr.extract_tire_info(unwrapped_image)
        return ocr_result
