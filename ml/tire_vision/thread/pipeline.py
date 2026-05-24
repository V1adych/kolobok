from dataclasses import dataclass
from typing import List, Optional
import time

import numpy as np
from fastapi import HTTPException

from models import Stud
from tire_vision.config import STUD_HEALTH_SCORES, STUD_VOLUMES, TireThreadPipelineConfig
from tire_vision.options import TireThreadPipelineOptions
from tire_vision.thread.segmentator.pipeline import ThreadSegmentator, TireInstance
from tire_vision.thread.studs.pipeline import StudPipeline
from tire_vision.thread.depth.model import DepthRegressor

import logging


@dataclass(frozen=True)
class AnalyzedTire:
    box: tuple[int, int, int, int]
    score: float
    depth: float
    studs: List[Stud]
    num_studs: int
    num_studs_classified: int
    fraction_healthy: float | None
    mask: np.ndarray


@dataclass(frozen=True)
class TireThreadPipelineResult:
    tires: List[AnalyzedTire]


class TireThreadPipeline:
    def __init__(self, config: TireThreadPipelineConfig):
        self.segmentator = ThreadSegmentator(config.thread_segmentator_config)
        self.stud_pipeline = StudPipeline(config.stud_pipeline_config)
        self.depth_regressor = DepthRegressor(config.depth_regressor_config)

        self.logger = logging.getLogger("tire_thread_pipeline")

    def _get_stud_matches(self, stud: Stud, tires: List[TireInstance]) -> List[int]:
        cx, cy, _, _ = stud.box
        matches = []
        for idx, tire in enumerate(tires):
            height, width = tire.mask.shape
            if 0 <= cx < width and 0 <= cy < height and tire.mask[cy, cx]:
                matches.append(idx)
        return matches

    def _group_studs(
        self,
        studs: List[Stud],
        tires: List[TireInstance],
        strategy: str,
    ) -> List[List[Stud]]:
        grouped = [[] for _ in tires]

        for stud in studs:
            matches = self._get_stud_matches(stud, tires)
            if len(matches) == 0:
                continue
            if len(matches) == 1:
                grouped[matches[0]].append(stud)
                continue

            if strategy == "all":
                for match_idx in matches:
                    grouped[match_idx].append(stud)
            elif strategy == "highest":
                best_idx = max(matches, key=lambda idx: tires[idx].score)
                grouped[best_idx].append(stud)

        return grouped

    def _fraction_healthy(self, studs: List[Stud]) -> tuple[int, float | None]:
        num_studs_classified = sum(STUD_VOLUMES[stud.label_id] for stud in studs)
        if num_studs_classified == 0:
            return 0, None

        fraction_healthy = sum(STUD_HEALTH_SCORES[stud.label_id] for stud in studs) / num_studs_classified
        return num_studs_classified, fraction_healthy

    def __call__(self, image: np.ndarray, options: Optional[TireThreadPipelineOptions] = None) -> TireThreadPipelineResult:
        self.logger.info("Starting tire thread pipeline")
        start_time = time.perf_counter()

        segmentator_options = options.thread_segmentator_options if options is not None else None
        stud_options = options.stud_pipeline_options if options is not None else None
        ambiguity_strategy = options.ambiguous_stud_resolution_strategy if options is not None else "highest"

        tires = self.segmentator(image, options=segmentator_options)
        if len(tires) == 0:
            self.logger.error("Tire not found on the image, or it is too small")
            raise HTTPException(status_code=500, detail="Tire not found on the image, or it is too small")

        studs, _, _ = self.stud_pipeline(image, options=stud_options)
        grouped_studs = self._group_studs(studs, tires, ambiguity_strategy)

        tire_results = []
        cropped_images = [self.segmentator.crop_tire(image, tire, options=segmentator_options) for tire in tires]
        resized_images = np.stack([self.depth_regressor.resize(image) for image in cropped_images if image is not None], axis=0)
        depths = self.depth_regressor(resized_images)
        for tire, tire_studs, cropped_image, depth in zip(tires, grouped_studs, cropped_images, depths):

            num_studs_classified, fraction_healthy = self._fraction_healthy(tire_studs)
            tire_results.append(
                AnalyzedTire(
                    box=tire.box,
                    score=tire.score,
                    depth=depth,
                    studs=tire_studs,
                    num_studs=len(tire_studs),
                    num_studs_classified=num_studs_classified,
                    fraction_healthy=fraction_healthy,
                    mask=tire.mask,
                )
            )

        if len(tire_results) == 0:
            self.logger.error("No valid tire crops were produced")
            raise HTTPException(status_code=500, detail="Tire not found on the image, or it is too small")

        latency = time.perf_counter() - start_time
        self.logger.info(f"Tire thread pipeline completed in {latency:.4f} seconds")
        self.logger.info(f"Detected {len(tire_results)} tires")
        return TireThreadPipelineResult(tires=tire_results)
