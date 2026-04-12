from dataclasses import dataclass
import logging
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from tire_vision.config import RTMDetSegmentatorConfig, ort_opts, ort_providers
from tire_vision.options import RTMDetSegmentatorOptions
from tire_vision.utils import nms


@dataclass(frozen=True)
class TireInstance:
    box_xyxy: Tuple[int, int, int, int]
    score: float
    mask: np.ndarray


@dataclass(frozen=True)
class RTMDetSegmentatorResult:
    image_size: Tuple[int, int]
    tires: List[TireInstance]


class RTMDetSegmentatorPipeline:
    def __init__(self, config: RTMDetSegmentatorConfig):
        self.config = config
        self.default_options = RTMDetSegmentatorOptions()
        self.det_session = ort.InferenceSession(config.detector_onnx, providers=ort_providers, sess_options=ort_opts)
        self.mask_session = ort.InferenceSession(config.mask_decoder_onnx, providers=ort_providers, sess_options=ort_opts)
        self.logger = logging.getLogger("rtmdet_segmentator_pipeline")

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
        options: RTMDetSegmentatorOptions,
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
        options: RTMDetSegmentatorOptions,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        keep = scores >= options.confidence_threshold
        return boxes[keep], scores[keep], kernels[keep], priors[keep]

    def _nms_filter(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        kernels: np.ndarray,
        priors: np.ndarray,
        options: RTMDetSegmentatorOptions,
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

    def forward(self, image: np.ndarray, options: Optional[RTMDetSegmentatorOptions] = None) -> RTMDetSegmentatorResult:
        start_time = time.perf_counter()
        opts = options if options is not None else self.default_options
        orig_h, orig_w = image.shape[:2]
        detector_inputs = self._preprocess(image)

        boxes, scores, kernels, priors, mask_feat = self.det_session.run(
            ["boxes", "scores", "kernels", "priors", "mask_feat"],
            {"input": detector_inputs},
        )
        boxes = boxes[0]
        scores = scores[0]
        kernels = kernels[0]
        priors = priors[0]

        boxes, scores, kernels, priors = self._global_topk(boxes, scores, kernels, priors, opts)
        boxes, scores, kernels, priors = self._confidence_filter(boxes, scores, kernels, priors, opts)
        boxes, scores, kernels, priors = self._nms_filter(boxes, scores, kernels, priors, opts)
        boxes, scores, kernels, priors = self._filter_invalid(boxes, scores, kernels, priors)

        num_instances = len(boxes)
        if num_instances == 0:
            return RTMDetSegmentatorResult(image_size=(orig_h, orig_w), tires=[])

        max_instances = self.config.max_mask_instances
        num_params = kernels.shape[-1]
        kernels_pad = np.zeros((1, max_instances, num_params), dtype=np.float32)
        priors_pad = np.zeros((1, max_instances, 4), dtype=np.float32)
        valid = np.zeros((1, max_instances), dtype=np.float32)
        kernels_pad[0, :num_instances] = kernels
        priors_pad[0, :num_instances] = priors
        valid[0, :num_instances] = 1.0

        (mask_probs,) = self.mask_session.run(
            ["mask_probs"],
            {
                "mask_feat": mask_feat.astype(np.float32),
                "kernels": kernels_pad,
                "priors": priors_pad,
                "valid": valid,
            },
        )

        scale_x = orig_w / self.config.resize_shape[0]
        scale_y = orig_h / self.config.resize_shape[1]
        tires = []
        for box, score, mask_prob in zip(boxes, scores, mask_probs[0, :num_instances]):
            x1, y1, x2, y2 = box
            box_xyxy = np.array([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], dtype=np.float32)
            box_xyxy[0::2] = np.clip(box_xyxy[0::2], 0, orig_w)
            box_xyxy[1::2] = np.clip(box_xyxy[1::2], 0, orig_h)
            mask = cv2.resize(mask_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) >= opts.mask_threshold
            tires.append(
                TireInstance(
                    box_xyxy=tuple(np.round(box_xyxy).astype(np.int32).tolist()),
                    score=float(score),
                    mask=mask,
                )
            )

        latency = time.perf_counter() - start_time
        self.logger.info(f"RTMDet segmentator pipeline completed in {latency:.4f} seconds")
        return RTMDetSegmentatorResult(image_size=(orig_h, orig_w), tires=tires)

    def __call__(self, image: np.ndarray, options: Optional[RTMDetSegmentatorOptions] = None) -> RTMDetSegmentatorResult:
        return self.forward(image, options=options)
