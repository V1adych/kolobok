from dataclasses import replace
from typing import Optional

import numpy as np
import cv2
from fastapi import HTTPException

from tire_vision.config import SidewallUnwrapperConfig
from tire_vision.options import SidewallUnwrapperOptions


import logging

class SidewallUnwrapper:
    def __init__(self, config: SidewallUnwrapperConfig):
        self.config = config
        self.logger = logging.getLogger("sidewall_unwrapper")
        clip_limit = self.config.clahe_clip_limit
        tile_grid_size = self.config.clahe_tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        self.logger.info("SidewallUnwrapper initialized successfully")

    def _postprocess_mask(self, mask: np.ndarray):
        kernel_size = self.config.mask_postprocess_ksize
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        processed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(processed_mask)
        if n <= 1:
            return processed_mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        return (labels == largest).astype(np.uint8) * 255

    def _ellipse_params_from_mask(self, mask_cc):
        cnts, _ = cv2.findContours(mask_cc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            msg = "Failed to find full sidewall on the image. Make sure you are providing an entire sidewall ring"
            self.logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)
        cnt = max(cnts, key=cv2.contourArea)

        return cv2.fitEllipse(cnt)

    def forward(self, image: np.ndarray, mask: np.ndarray, options: Optional[SidewallUnwrapperOptions] = None):
        h, w = image.shape[:2]
        if options is not None:
            self.config = replace(self.config, options=options)
        mask_cc = self._postprocess_mask(mask)
        (x, y), (major_axis, minor_axis), angle = self._ellipse_params_from_mask(mask_cc)
        r_minor = minor_axis / 2.0

        do_rectify = (major_axis / minor_axis) > self.config.rectify_aspect_ratio_threshold

        if do_rectify:
            rot = cv2.getRotationMatrix2D((x, y), angle, 1.0)
            image = cv2.warpAffine(image, rot, (w, h), flags=cv2.INTER_CUBIC)

            scale_y = minor_axis / major_axis
            scale = np.array([[1, 0, 0], [0, scale_y, y * (1 - scale_y)]], dtype=np.float32)
            image = cv2.warpAffine(image, scale, (w, h), flags=cv2.INTER_CUBIC)

        dst_size = self.config.options.polar_unwrap_size
        polar_image = cv2.warpPolar(image, dst_size, (x, y), r_minor, flags=cv2.INTER_CUBIC)

        polar_image = cv2.rotate(polar_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.config.concat_strip:
            _, w, _ = polar_image.shape
            strip_slide = np.concatenate([polar_image[:, w // 2 :], polar_image[:, : w // 2]], axis=1)
            polar_image = np.concatenate([polar_image, strip_slide], axis=0)

        lab_image = cv2.cvtColor(polar_image, cv2.COLOR_RGB2LAB)
        lab_image[:, :, 0] = self.clahe.apply(lab_image[:, :, 0])

        return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    def __call__(self, image: np.ndarray, mask: np.ndarray, options: Optional[SidewallUnwrapperOptions] = None):
        return self.forward(image, mask, options=options)
