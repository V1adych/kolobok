from dataclasses import dataclass
import logging
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tire_vision.config import ThreadSegmentatorConfig
from tire_vision.options import ThreadSegmentatorOptions
from tire_vision.thread.segmentator.model import ThreadSegmentatorModel
from tire_vision.utils import nms, xyxy2cxcywh, expit


@dataclass(frozen=True)
class TireInstance:
    box: Tuple[int, int, int, int]
    score: float
    mask: np.ndarray


class ThreadSegmentator:
    def __init__(self, config: ThreadSegmentatorConfig):
        self.config = config
        self.default_options = ThreadSegmentatorOptions()
        self.model = ThreadSegmentatorModel(config)
        self.logger = logging.getLogger("thread_segmentator")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(image_bgr, self.config.resize_shape, interpolation=cv2.INTER_LINEAR)
        return resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    def _global_topk(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        kernels: np.ndarray,
        priors: np.ndarray,
        options: ThreadSegmentatorOptions,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_classes = scores.shape[1]
        scores_flat = scores.reshape(-1)
        num_topk = min(options.pre_topk, scores_flat.size)
        topk_indices = np.argpartition(scores_flat, -num_topk)[-num_topk:]
        selected_scores = scores_flat[topk_indices]
        order = np.argsort(selected_scores)[::-1]
        topk_indices = topk_indices[order]
        selected_scores = selected_scores[order]
        box_indices = topk_indices // num_classes
        return boxes[box_indices], selected_scores, kernels[box_indices], priors[box_indices]

    def _confidence_filter(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        kernels: np.ndarray,
        priors: np.ndarray,
        options: ThreadSegmentatorOptions,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        keep = scores >= options.confidence_threshold
        return boxes[keep], scores[keep], kernels[keep], priors[keep]

    def _nms_filter(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        kernels: np.ndarray,
        priors: np.ndarray,
        options: ThreadSegmentatorOptions,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        keep = nms(boxes, scores, options.nms_iou_threshold)[: self.config.max_mask_instances]
        return boxes[keep], scores[keep], kernels[keep], priors[keep]

    def _filter_invalid(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        kernels: np.ndarray,
        priors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[keep], scores[keep], kernels[keep], priors[keep]

    def _decode_masks(
        self,
        mask_feat: np.ndarray,
        kernels: np.ndarray,
        priors: np.ndarray,
    ) -> np.ndarray:
        num_instances = len(kernels)
        num_params = kernels.shape[-1]
        kernels_pad = np.zeros((1, self.config.max_mask_instances, num_params), dtype=np.float32)
        priors_pad = np.zeros((1, self.config.max_mask_instances, 4), dtype=np.float32)
        valid = np.zeros((1, self.config.max_mask_instances), dtype=np.float32)
        kernels_pad[0, :num_instances] = kernels
        priors_pad[0, :num_instances] = priors
        valid[0, :num_instances] = 1.0
        mask_logits = self.model.decode_masks(mask_feat, kernels_pad, priors_pad, valid)
        return mask_logits[0, :num_instances]

    def _to_tire_instances(
        self,
        image_shape: Tuple[int, int],
        boxes: np.ndarray,
        scores: np.ndarray,
        mask_logits: np.ndarray,
        options: ThreadSegmentatorOptions,
    ) -> List[TireInstance]:
        orig_h, orig_w = image_shape
        scale_x = orig_w / self.config.resize_shape[0]
        scale_y = orig_h / self.config.resize_shape[1]
        tires = []

        for box, score, mask_logit in zip(boxes, scores, mask_logits):
            box_xyxy = np.array([box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y], dtype=np.float32)
            box_xyxy[0::2] = np.clip(box_xyxy[0::2], 0, orig_w)
            box_xyxy[1::2] = np.clip(box_xyxy[1::2], 0, orig_h)

            logits_resized = cv2.resize(mask_logit, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            probs = expit(logits_resized)
            mask = probs >= options.mask_threshold
            if int(np.count_nonzero(mask)) < options.min_tire_pixels:
                continue

            box_cxcywh = xyxy2cxcywh(box_xyxy[None])[0]
            tires.append(
                TireInstance(
                    box=tuple(np.round(box_cxcywh).astype(np.int32).tolist()),
                    score=float(score),
                    mask=mask,
                )
            )

        return tires

    def forward(self, image: np.ndarray, options: Optional[ThreadSegmentatorOptions] = None) -> List[TireInstance]:
        start_time = time.perf_counter()
        opts = options if options is not None else self.default_options
        boxes, scores, kernels, priors, mask_feat = self.model.detect(self._preprocess(image))
        boxes = boxes[0]
        scores = scores[0]
        kernels = kernels[0]
        priors = priors[0]

        boxes, scores, kernels, priors = self._global_topk(boxes, scores, kernels, priors, opts)
        boxes, scores, kernels, priors = self._confidence_filter(boxes, scores, kernels, priors, opts)
        boxes, scores, kernels, priors = self._nms_filter(boxes, scores, kernels, priors, opts)
        boxes, scores, kernels, priors = self._filter_invalid(boxes, scores, kernels, priors)
        if len(boxes) == 0:
            return []

        tires = self._to_tire_instances(image.shape[:2], boxes, scores, self._decode_masks(mask_feat, kernels, priors), opts)
        latency = time.perf_counter() - start_time
        self.logger.info(f"Thread segmentation completed in {latency:.4f} seconds")
        return tires

    def crop_tire(self, image: np.ndarray, tire: TireInstance, options: Optional[ThreadSegmentatorOptions] = None) -> Optional[np.ndarray]:
        opts = options if options is not None else self.default_options
        mask = tire.mask[..., None].astype(np.uint8)
        if np.count_nonzero(mask) < opts.min_tire_pixels:
            return None

        background = np.full_like(image, 255)
        image_masked = (image * mask) + (background * (1 - mask))
        coords = np.where(mask[..., 0] > 0)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        height, width = image.shape[:2]
        pad_h = int(height * opts.padding_frac)
        pad_w = int(width * opts.padding_frac)
        y_min = max(0, y_min - pad_h)
        y_max = min(height, y_max + pad_h)
        x_min = max(0, x_min - pad_w)
        x_max = min(width, x_max + pad_w)
        return image_masked[y_min:y_max, x_min:x_max]

    def __call__(self, image: np.ndarray, options: Optional[ThreadSegmentatorOptions] = None) -> List[TireInstance]:
        return self.forward(image, options=options)
