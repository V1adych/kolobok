from typing import List, Dict, Any, Tuple
import time

import numpy as np
import cv2

from tire_vision.config import SpikePipelineConfig
from tire_vision.segmentation.onnx import OnnxSegmentator
from tire_vision.thread.spikes.classifier import SpikeClassifier

import logging


class SpikePipeline:
    def __init__(self, config: SpikePipelineConfig):
        self.config = config
        self.segmentator = OnnxSegmentator(
            self.config.spike_segmentator_onnx,
            self.config.resize_shape,
            self.config.confidence_threshold,
            self.config.resize_shape,
        )

        self.classifier = SpikeClassifier(self.config.spike_classifier_onnx)

        self.kernel = np.ones(shape=(3, 3), dtype=np.uint8)

        self.logger = logging.getLogger("spikes")

        self.logger.info("Spike Pipeline module initialized")

    def _apply_morphology(self, mask: np.ndarray):
        mask = cv2.erode(mask, self.kernel, iterations=self.config.erosion_iterations)
        mask = cv2.dilate(mask, self.kernel, iterations=self.config.dilation_iterations)
        return mask

    def _get_crop(
        self,
        image: np.ndarray,
        c1: int,
        c2: int,
        crop_half: int,
    ) -> np.ndarray:
        image_height, image_width, num_channels = image.shape
        crop_size = crop_half * 2

        x1_clipped = max(c1 - crop_half, 0)
        y1_clipped = max(c2 - crop_half, 0)
        x2_clipped = min(c1 + crop_half, image_width)
        y2_clipped = min(c2 + crop_half, image_height)

        left_pad = max(0, crop_half - c1)
        top_pad = max(0, crop_half - c2)

        destination = np.zeros((crop_size, crop_size, num_channels), dtype=image.dtype)
        source = image[y1_clipped:y2_clipped, x1_clipped:x2_clipped, :]

        destination[
            top_pad : top_pad + source.shape[0],
            left_pad : left_pad + source.shape[1],
            :,
        ] = source

        return destination

    def _get_spike_crops(
        self, image: np.ndarray, detection_mask: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        _, _, _, centroids = cv2.connectedComponentsWithStats(
            detection_mask, connectivity=8, ltype=cv2.CV_32S
        )
        centroids = centroids[1:]

        crops = []
        crop_size = self.config.crop_size
        crop_half = crop_size // 2
        for c1, c2 in centroids:
            spike_image = self._get_crop(image, int(c1), int(c2), crop_half)
            crops.append(spike_image)

        return crops, centroids

    def __call__(self, image: np.ndarray) -> List[Dict[str, Any]]:
        h, w, _ = image.shape
        self.logger.info("Running spike detection")
        start_time = time.perf_counter()

        image_resized = cv2.resize(
            image, self.config.resize_shape, interpolation=cv2.INTER_LINEAR
        )

        detection_mask = self.segmentator(image_resized)
        detection_mask = self._apply_morphology(detection_mask)

        crops, centroids = self._get_spike_crops(image_resized, detection_mask)

        centroids[:, 0] *= w / self.config.resize_shape[1]
        centroids[:, 1] *= h / self.config.resize_shape[0]
        centroids = centroids.astype(np.int32)

        boxes = [
            {
                "box": (
                    int(c1),
                    int(c2),
                    self.config.crop_size,
                    self.config.crop_size,
                )
            }
            for c1, c2 in centroids
        ]

        spikes = []
        if len(crops) > 0:
            crops = np.stack(crops).astype(np.float32).transpose(0, 3, 1, 2) / 255

            spike_classes = self.classifier(crops).argmax(axis=1)

            for box, spike_class in zip(boxes, spike_classes):
                if spike_class == 2:
                    continue
                spikes.append({**box, "class": int(spike_class)})

        latency = time.perf_counter() - start_time
        self.logger.info(f"Spike detection completed in {latency:.4f} seconds")

        return spikes
