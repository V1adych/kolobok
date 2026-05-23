from typing import List, Tuple, Optional
import time

import numpy as np
import cv2
import onnxruntime as ort

from tire_vision.config import StudPipelineConfig, ort_providers, ort_opts, STUD_LABELS, STUD_VOLUMES, STUD_HEALTH_SCORES
from tire_vision.options import StudPipelineOptions
from tire_vision.utils import cxcywh2xyxy, nms, xyxy2cxcywh, expit
from models import Stud

import logging

class StudPipeline:
    def __init__(self, config: StudPipelineConfig):
        self.config = config
        self.default_options = StudPipelineOptions()
        self.det_session = ort.InferenceSession(self.config.spike_detector_onnx, providers=ort_providers, sess_options=ort_opts)

        self.logger = logging.getLogger("stud_pipeline")

        self.logger.info("StudPipeline initialized successfully!")

    def _global_topk(self, boxes: np.ndarray, logits: np.ndarray, options: StudPipelineOptions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, num_classes = logits.shape
        logits_flat = logits.reshape(-1)
        topk_indices = np.argpartition(logits_flat, -options.max_detections)[-options.max_detections :]
        logits_selected = logits_flat[topk_indices]
        ids = np.argsort(logits_selected)[::-1]
        logits_selected = logits_selected[ids]
        topk_indices = topk_indices[ids]
        box_indices = topk_indices // num_classes
        labels_selected = topk_indices % num_classes
        boxes_selected = boxes[box_indices]

        return boxes_selected, logits_selected, labels_selected

    def _nms_filter(self, boxes: np.ndarray, logits: np.ndarray, labels: np.ndarray, options: StudPipelineOptions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep_ids = nms(boxes, logits, options.nms_iou_threshold)

        return boxes[keep_ids], logits[keep_ids], labels[keep_ids]

    def _confidence_filter(self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, options: StudPipelineOptions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep = scores > options.confidence_threshold

        return boxes[keep], scores[keep], labels[keep]

    def _filter_invalid(self, boxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        keep = ((boxes[:, 2] - boxes[:, 0]) > 0) & ((boxes[:, 3] - boxes[:, 1]) > 0) & (labels > 0)

        return boxes[keep], labels[keep]

    def _filter_farthest(self, boxes_cxcywh: np.ndarray, labels: np.ndarray, options: StudPipelineOptions) -> Tuple[np.ndarray, np.ndarray]:
        if len(boxes_cxcywh) == 0:
            return boxes_cxcywh, labels

        avg_position = np.mean(boxes_cxcywh, axis=0, keepdims=True)[:, :2]
        distance_sq = np.sum(np.square(boxes_cxcywh[:, :2] - avg_position), axis=1)
        threshold = np.quantile(distance_sq, 1 - options.filter_frac)
        keep = distance_sq <= threshold

        return boxes_cxcywh[keep], labels[keep]

    def __call__(self, image: np.ndarray, options: Optional[StudPipelineOptions] = None) -> List[Stud]:
        start_time = time.perf_counter()
        opts = options if options is not None else self.default_options

        h, w, _ = image.shape

        image = cv2.resize(image, self.config.resize_shape, interpolation=cv2.INTER_LINEAR)
        image = image.transpose(2, 0, 1)[None].astype(np.float32) / 255

        boxes_cxcywh, logits = self.det_session.run(None, {"input": image})
        boxes_cxcywh = boxes_cxcywh.squeeze(0)
        logits = logits.squeeze(0)

        boxes_cxcywh, logits, labels = self._global_topk(boxes_cxcywh, logits, opts)
        boxes_xyxy = cxcywh2xyxy(boxes_cxcywh)

        scores = expit(logits)
        boxes_xyxy, scores, labels = self._confidence_filter(boxes_xyxy, scores, labels, opts)
        boxes_xyxy, scores, labels = self._nms_filter(boxes_xyxy, scores, labels, opts)
        boxes_xyxy, labels = self._filter_invalid(boxes_xyxy, labels)

        boxes_cxcywh = xyxy2cxcywh(boxes_xyxy)
        boxes_cxcywh, labels = self._filter_farthest(boxes_cxcywh, labels, opts)

        boxes_cxcywh = boxes_cxcywh * np.array([w, h, w, h], dtype=np.float32)
        boxes_cxcywh = boxes_cxcywh.astype(np.int32)

        labels = labels - 1
        studs = list(map(lambda box, label_id: Stud(box=box, label_id=label_id, label=STUD_LABELS[label_id]), boxes_cxcywh.tolist(), labels.tolist()))
        num_studs_classified = sum(map(lambda stud: STUD_VOLUMES[stud.label_id], studs))
        fraction_healthy = 0
        if num_studs_classified > 0:
            fraction_healthy = sum(map(lambda stud: STUD_HEALTH_SCORES[stud.label_id], studs)) / num_studs_classified

        latency = time.perf_counter() - start_time
        self.logger.info(f"Stud pipeline completed in {latency:.4f} seconds")

        return studs, num_studs_classified, fraction_healthy
