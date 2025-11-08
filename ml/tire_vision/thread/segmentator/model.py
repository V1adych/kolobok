from typing import Optional
from dataclasses import replace
import logging
import time

import numpy as np


from tire_vision.config import ThreadSegmentatorConfig
from tire_vision.segmentation.onnx import OnnxSegmentator
from tire_vision.options import ThreadSegmentatorOptions


class ThreadSegmentator:
    def __init__(self, config: ThreadSegmentatorConfig):
        self.config = config
        self.logger = logging.getLogger("thread_segmentator")

        self.segmentator = OnnxSegmentator(
            self.config.thread_segmentator_onnx,
            self.config.resize_shape,
        )

        self.logger.info("ThreadSegmentator initialized successfully")

    def forward(self, image: np.ndarray, options: Optional[ThreadSegmentatorOptions] = None):
        start_time = time.perf_counter()
        if options is not None:
            self.config = replace(self.config, options=options)

        mask = self.segmentator(image, threshold=self.config.options.confidence_threshold)

        end_time = time.perf_counter()
        self.logger.info(f"Completed thread segmentation in {end_time - start_time} seconds")

        return mask

    def crop_tire(self, image: np.ndarray, options: Optional[ThreadSegmentatorOptions] = None):
        self.logger.info("Cropping tire")
        if options is not None:
            self.config = replace(self.config, options=options)
        mask = self.forward(image)[..., None] // 255

        if np.count_nonzero(mask) < self.config.options.min_tire_pixels:
            self.logger.warning("Tire not found on the image, or it is too small")
            return None

        background = np.full_like(image, 255)

        image_masked = (image * mask) + (background * (1 - mask))

        coords = np.where(mask > 0)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        height, width = image.shape[:2]
        pad_h = int(height * self.config.options.padding_frac)
        pad_w = int(width * self.config.options.padding_frac)

        y_min = max(0, y_min - pad_h)
        y_max = min(height, y_max + pad_h)
        x_min = max(0, x_min - pad_w)
        x_max = min(width, x_max + pad_w)

        cropped_image = image_masked[y_min:y_max, x_min:x_max]

        self.logger.info(f"Cropped image shape: {cropped_image.shape}")

        return cropped_image
