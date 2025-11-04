import logging
import time

import numpy as np

from tire_vision.config import SidewallSegmentatorConfig
from tire_vision.segmentation.onnx import OnnxSegmentator
from tire_vision.options import SidewallSegmentatorOptions
from dataclasses import replace


class SidewallSegmentator:
    def __init__(self, config: SidewallSegmentatorConfig):
        self.config = config
        self.logger = logging.getLogger("sidewall_segmentator")

        self.segmentator = OnnxSegmentator(
            self.config.sidewall_segmentator_onnx,
            self.config.resize_shape,
        )

        self.logger.info("SidewallSegmentator initialized successfully")

    def forward(
        self, image: np.ndarray, options: SidewallSegmentatorOptions | None = None
    ):
        start_time = time.perf_counter()
        if options is not None:
            self.config = replace(self.config, options=options)
        mask = self.segmentator(
            image, threshold=self.config.options.confidence_threshold
        )

        end_time = time.perf_counter()
        self.logger.info(
            f"Completed sidewall segmentation in {end_time - start_time} seconds"
        )

        return mask

    def __call__(
        self, image: np.ndarray, options: SidewallSegmentatorOptions | None = None
    ):
        return self.forward(image, options=options)
