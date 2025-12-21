from typing import List, Tuple, Optional
from dataclasses import replace
import time

import numpy as np
import cv2
import onnxruntime as ort

from tire_vision.config import StudPipelineConfig, ort_providers, ort_opts, META_MAPPING, META_TO_LABEL_MAPPING, LABEL_MAPPING
from tire_vision.options import StudPipelineOptions
from models import Stud

import logging


def cxcywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def xyxy2cxcywh(boxes: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = boxes.T
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.stack([cx, cy, w, h], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int32)

    boxes = boxes.astype(np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    areas = w * h

    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih

        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-7)

        inds = np.where(iou <= iou_threshold)[0]
        order = rest[inds]

    return np.array(keep, dtype=np.int32)


class StudPipeline:
    def __init__(self, config: StudPipelineConfig):
        self.config = config
        self.det_session = ort.InferenceSession(self.config.spike_detector_onnx, providers=ort_providers, sess_options=ort_opts)

        self.logger = logging.getLogger("stud_pipeline")

        self.logger.info("StudPipeline initialized successfully!")

    def _global_topk(self, boxes: np.ndarray, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, num_classes = logits.shape
        logits_flat = logits.reshape(-1)
        topk_indices = np.argpartition(logits_flat, -self.config.options.max_detections)[-self.config.options.max_detections :]
        logits_selected = logits_flat[topk_indices]
        ids = np.argsort(logits_selected)[::-1]
        logits_selected = logits_selected[ids]
        topk_indices = topk_indices[ids]
        box_indices = topk_indices // num_classes
        labels_selected = topk_indices % num_classes
        boxes_selected = boxes[box_indices]

        return boxes_selected, logits_selected, labels_selected

    def _nms_filter(self, boxes: np.ndarray, logits: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep_ids = nms(boxes, logits, self.config.options.nms_iou_threshold)

        return boxes[keep_ids], logits[keep_ids], labels[keep_ids]

    def _confidence_filter(self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep = scores > self.config.options.confidence_threshold

        return boxes[keep], scores[keep], labels[keep]

    def _filter_invalid(self, boxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        keep = ((boxes[:, 2] - boxes[:, 0]) > 0) & ((boxes[:, 3] - boxes[:, 1]) > 0) & (labels > 0)

        return boxes[keep], labels[keep]

    def __call__(self, image: np.ndarray, options: Optional[StudPipelineOptions] = None) -> List[Stud]:
        start_time = time.perf_counter()
        if options is not None:
            self.config = replace(self.config, options=options)

        h, w, _ = image.shape

        image = cv2.resize(image, self.config.resize_shape, interpolation=cv2.INTER_LINEAR)
        image = image.transpose(2, 0, 1)[None].astype(np.float32) / 255

        boxes_cxcywh, logits = self.det_session.run(None, {"input": image})
        boxes_cxcywh = boxes_cxcywh.squeeze(0)
        logits = logits.squeeze(0)

        boxes_cxcywh, logits, labels = self._global_topk(boxes_cxcywh, logits)
        boxes_xyxy = cxcywh2xyxy(boxes_cxcywh)

        scores = 1.0 / (1.0 + np.exp(-logits))
        boxes_xyxy, scores, labels = self._confidence_filter(boxes_xyxy, scores, labels)
        boxes_xyxy, scores, labels = self._nms_filter(boxes_xyxy, scores, labels)
        boxes_xyxy, labels = self._filter_invalid(boxes_xyxy, labels)

        boxes_cxcywh = xyxy2cxcywh(boxes_xyxy)
        boxes_cxcywh = boxes_cxcywh * np.array([w, h, w, h], dtype=np.float32)
        boxes_cxcywh = boxes_cxcywh.astype(np.int32)

        labels = labels - 1

        def to_stud(box: Tuple[int, int, int, int], meta_label_id: int) -> Stud:
            meta_label = META_MAPPING[meta_label_id]
            label_id = META_TO_LABEL_MAPPING[meta_label_id]
            label = LABEL_MAPPING[label_id]
            return Stud(box=box, label_id=label_id, label=label, meta_label=meta_label, meta_label_id=meta_label_id)

        result = list(map(to_stud, boxes_cxcywh.tolist(), labels.tolist()))

        latency = time.perf_counter() - start_time
        self.logger.info(f"Stud pipeline completed in {latency:.4f} seconds")

        return result
