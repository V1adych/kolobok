from typing import Optional
from dataclasses import replace
import time

import numpy as np
from fastapi import HTTPException

from tire_vision.thread.segmentator.model import ThreadSegmentator
from tire_vision.thread.studs.pipeline import StudPipeline
from tire_vision.thread.depth.model import DepthRegressor
from tire_vision.config import TireThreadPipelineConfig
from tire_vision.options import TireThreadPipelineOptions
from models import TireThreadPipelineResult

import logging


class TireThreadPipeline:
    def __init__(self, config: TireThreadPipelineConfig):
        self.segmentator = ThreadSegmentator(config.thread_segmentator_config)
        self.stud_pipeline = StudPipeline(config.stud_pipeline_config)
        self.depth_regressor = DepthRegressor(config.depth_regressor_config)

        self.logger = logging.getLogger("tire_thread_pipeline")

    def __call__(
        self, image: np.ndarray, options: Optional[TireThreadPipelineOptions] = None
    ) -> TireThreadPipelineResult:
        self.logger.info("Starting tire thread pipeline")
        start_time = time.perf_counter()
        if options is not None:
            self.segmentator.config = replace(self.segmentator.config, options=options.thread_segmentator_options)
            self.stud_pipeline.config = replace(self.stud_pipeline.config, options=options.stud_pipeline_options)

        cropped_image = self.segmentator.crop_tire(image)
        if cropped_image is None:
            self.logger.error("Tire not found on the image, or it is too small")
            raise HTTPException(status_code=500, detail="Tire not found on the image, or it is too small")

        studs = self.stud_pipeline(image)
        fraction_healthy = None
        if len(studs) > 0:
            fraction_healthy = np.mean(list(map(lambda stud: stud.label_id == 1, studs)))

        depth = self.depth_regressor(cropped_image)

        latency = time.perf_counter() - start_time
        self.logger.info(f"Tire thread pipeline completed in {latency:.4f} seconds")

        result = TireThreadPipelineResult(depth=depth, studs=studs, fraction_healthy=fraction_healthy)
        self.logger.info(f"Cropped image shape: {cropped_image.shape} Depth: {depth} Number of studs: {len(studs)}")

        return result
