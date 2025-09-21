from dataclasses import dataclass
from pathlib import Path
import json
import base64
from typing import List, Dict, Optional
import requests
import logging
from tqdm import tqdm
import shutil
from PIL import Image
import io

import tyro
import numpy as np
import polars as pl


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Args:
    test_root: str
    annot_name: str = "_annotations.coco.json"
    url: str = "http://localhost:8000/api/v1/analyze_thread"
    token: str = "kolobok_token"
    img_save_dir: Optional[str] = None
    iou_threshold: float = 0.2


def get_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_predictions(url: str, image_path: str, token: str) -> List[Dict]:
    image_base64 = get_image_base64(image_path)
    response = requests.post(
        url, json={"image": image_base64}, headers={"Authorization": f"Bearer {token}"}
    )

    if response.status_code != 200:
        logging.error(
            f"Error getting predictions for {image_path}: {response.status_code}"
        )
        return None

    return response.json()


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


def get_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    boxes1_ = boxes1[:, None, :]  # [N, 1, 4]
    boxes2_ = boxes2[None, :, :]  # [1, M, 4]

    # Calculate intersection coordinates
    x1 = np.maximum(boxes1_[..., 0], boxes2_[..., 0])  # [N, M]
    y1 = np.maximum(boxes1_[..., 1], boxes2_[..., 1])  # [N, M]
    x2 = np.minimum(boxes1_[..., 2], boxes2_[..., 2])  # [N, M]
    y2 = np.minimum(boxes1_[..., 3], boxes2_[..., 3])  # [N, M]

    # Calculate intersection area
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)  # [N, M]

    # Calculate areas of both sets of boxes
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # [N]
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # [M]

    # Broadcast areas for union calculation
    area1 = area1[:, None]  # [N, 1]
    area2 = area2[None, :]  # [1, M]

    # Calculate union
    union = area1 + area2 - intersection  # [N, M]

    # Calculate IoU, avoiding division by zero
    iou = np.where(union > 0, intersection / union, 0.0)

    return iou


def calculate_metrics(
    boxes_pred: np.ndarray,
    labels_pred: np.ndarray,
    boxes_gt: np.ndarray,
    labels_gt: np.ndarray,
    iou_threshold: float,
) -> Dict:
    iou_mat = get_iou_matrix(boxes_pred, boxes_gt)
    metrics = {
        "det_tp": 0,
        "det_fp": 0,
        "det_fn": 0,
        "cls_correct": 0,
        "cls_incorrect": 0,
    }
    detected = np.zeros(boxes_gt.shape[0], dtype=bool)
    for pred_idx in range(iou_mat.shape[0]):
        gt_idx = np.argmax(iou_mat[pred_idx])
        best_iou = iou_mat[pred_idx, gt_idx]
        if best_iou >= iou_threshold:
            detected[gt_idx] = True
            iou_mat[:, gt_idx] = -np.inf
            metrics["det_tp"] += 1
            if labels_pred[pred_idx] == labels_gt[gt_idx]:
                metrics["cls_correct"] += 1
            else:
                metrics["cls_incorrect"] += 1
        else:
            metrics["det_fp"] += 1
    metrics["det_fn"] = int(np.sum(~detected))

    return metrics

def base64_to_np_array(base64_str: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(base64_str))))


def main():
    args = tyro.cli(Args)

    test_root = Path(args.test_root)
    annot_path = test_root / args.annot_name

    with open(annot_path, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    all_metrics = []
    for image in tqdm(images):
        image_path = test_root / image["file_name"]
        image_id = image["id"]
        annotations_for_image = [a for a in annotations if a["image_id"] == image_id]

        predictions = get_predictions(args.url, image_path, args.token)
        perf_stats, spikes = predictions["perf_stats"], predictions["spikes"]
        image_np = base64_to_np_array(predictions["image"])

        boxes_pred = cxcywh_to_xyxy(
            np.array([spike["box"] for spike in spikes], dtype=np.float32)
        )
        labels_pred = np.array([spike["class"] for spike in spikes], dtype=np.int32)
        boxes_annot = cxcywh_to_xyxy(
            np.array(
                [annotation["bbox"] for annotation in annotations_for_image],
                dtype=np.float32,
            )
        )
        labels_annot = np.array(
            [annotation["category_id"] - 1 for annotation in annotations_for_image],
            dtype=np.int32,
        )

        metrics = calculate_metrics(boxes_pred, labels_pred, boxes_annot, labels_annot, args.iou_threshold)
        if args.img_save_dir is not None:
            img_save_path = Path(args.img_save_dir) / f"{image_id}.png"
            img_save_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image_np).save(img_save_path)
            metrics["img_save_path"] = str(img_save_path)
        all_metrics.append(metrics)
    
    df = pl.DataFrame(all_metrics)
    print(df)
    df.write_csv(Path(args.img_save_dir) / "metrics.csv")
    print(df.select("det_tp", "det_fp", "det_fn", "cls_correct", "cls_incorrect").mean())



if __name__ == "__main__":
    main()
