from dataclasses import dataclass, field
from typing import Tuple, List, Dict
from pathlib import Path
import json

import numpy as np
import pandas as pd
import cv2
import tyro

from tqdm import tqdm

from tire_vision.thread.spikes.pipeline import SpikePipeline
from tire_vision.config import SpikePipelineConfig


@dataclass
class Args:
    det_images_dir: str
    det_masks_dir: str
    all_images_dir: str
    annotations_path: str
    dest_dir: str

    classes: Dict[str, int] = field(
        default_factory=lambda: {
            "normal": 0,
            "broken": 1,
            "absent": 1,
            "renewed": 0,
            "floating": 0,
            "false_positive": 2,
        }
    )


def get_spike_crop(
    image: np.ndarray, cx: int, cy: int, crop_size: int = 64
) -> np.ndarray:
    r = crop_size // 2
    x1 = np.clip(cx - r, 0, image.shape[1])
    y1 = np.clip(cy - r, 0, image.shape[0])
    x2 = np.clip(cx + r, 0, image.shape[1])
    y2 = np.clip(cy + r, 0, image.shape[0])

    cropped_image = image[y1:y2, x1:x2, :]

    # pad to crop_size
    if cropped_image.shape[0] != crop_size:
        if y1 != 0:
            cropped_image = np.pad(
                cropped_image,
                ((crop_size - cropped_image.shape[0], 0), (0, 0), (0, 0)),
                mode="constant",
            )
        else:
            cropped_image = np.pad(
                cropped_image,
                ((0, crop_size - cropped_image.shape[0]), (0, 0), (0, 0)),
                mode="constant",
            )

    if cropped_image.shape[1] != crop_size:
        if x1 != 0:
            cropped_image = np.pad(
                cropped_image,
                ((0, 0), (crop_size - cropped_image.shape[1], 0), (0, 0)),
                mode="constant",
            )
        else:
            cropped_image = np.pad(
                cropped_image,
                ((0, 0), (crop_size - cropped_image.shape[1], 0), (0, 0)),
                mode="constant",
            )
    return cropped_image


def create_dataset_from_labels(
    images_dir: Path,
    annotations_path: Path,
    dest_dir: Path,
    class2idx: Dict[str, int],
    counter: int = 0,
):
    with open(annotations_path, "r") as f:
        data = json.load(f)
    labels = []

    for item in data:
        image_name = Path(item["data"]["image"]).name.split("/")[-1]
        image_path = images_dir / image_name
        image = cv2.imread(str(image_path))

        image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

        if len(item["annotations"]) != 1:
            print(f"Expected 1 annotation, got {len(item['annotations'])}")
            continue

        for annot in item["annotations"][0]["result"]:
            if "value" not in annot:
                continue

            polygon = np.array(annot["value"]["points"], dtype=np.float32) * 5.12

            moments = cv2.moments(polygon)
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            spike_image = get_spike_crop(image_resized, cx, cy)
            spike_label = annot["value"]["polygonlabels"][0]
            spike_label_idx = class2idx[annot["value"]["polygonlabels"][0]]

            save_path = dest_dir / "images" / f"{counter:06d}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(save_path), spike_image)

            labels.append(
                {
                    "image_name": str(save_path.name),
                    "label": spike_label,
                    "label_id": spike_label_idx,
                }
            )

            counter += 1

    df = pd.DataFrame(labels)

    return df, counter


def create_dataset_from_detections(
    pipeline: SpikePipeline,
    images_dir: Path,
    masks_dir: Path,
    dest_dir: Path,
    class2idx: Dict[str, int],
    counter: int = 0,
):
    init_counter = counter
    all_imgs = list(images_dir.iterdir())
    labels = []

    loop = tqdm(all_imgs, desc="Processing images")

    for img_path in loop:
        img_name = img_path.name
        mask_path = masks_dir / img_name

        image = cv2.imread(str(img_path))
        gt_mask = (
            cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        )

        detection_mask = pipeline.segmentator(image)
        crops, centroids = pipeline._get_spike_crops(image, detection_mask)
        centroids[:, 0] *= image.shape[1] / pipeline.config.resize_shape[1]
        centroids[:, 1] *= image.shape[0] / pipeline.config.resize_shape[0]
        centroids = centroids.astype(np.int32)

        for crop, (c1, c2) in zip(crops, centroids):
            c1, c2 = int(c1), int(c2)
            x1 = np.clip(c1 - 32, 0, image.shape[1])
            y1 = np.clip(c2 - 32, 0, image.shape[0])
            x2 = np.clip(c1 + 32, 0, image.shape[1])
            y2 = np.clip(c2 + 32, 0, image.shape[0])
            mask_crop = gt_mask[y1:y2, x1:x2]

            if np.mean(mask_crop) > 0.01:
                continue

            save_path = dest_dir / "images" / f"{counter:06d}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), crop)

            labels.append(
                {
                    "image_name": str(save_path.name),
                    "label": "false_positive",
                    "label_id": class2idx["false_positive"],
                }
            )
            counter += 1

        loop.set_postfix(num_spikes=counter - init_counter)

    df = pd.DataFrame(labels)
    return df, counter


def main():
    args = tyro.cli(Args)

    det_images_dir = Path(args.det_images_dir)
    det_masks_dir = Path(args.det_masks_dir)
    all_images_dir = Path(args.all_images_dir)
    dest_dir = Path(args.dest_dir)
    annotations_path = Path(args.annotations_path)

    class2idx = args.classes

    spike_pipeline = SpikePipeline(config=SpikePipelineConfig(confidence_threshold=0.25, crop_size=64))

    df1, counter = create_dataset_from_labels(
        all_images_dir, annotations_path, dest_dir, class2idx, counter=0
    )
    df2, counter = create_dataset_from_detections(
        spike_pipeline,
        det_images_dir,
        det_masks_dir,
        dest_dir,
        class2idx,
        counter=counter,
    )

    df = pd.concat([df1, df2], axis=0)

    print("Saving labels to", dest_dir / "labels.csv")
    print("Total images:", len(df))
    print("Classes value counts:", df["label"].value_counts())
    df.to_csv(dest_dir / "labels.csv", index=False)


if __name__ == "__main__":
    main()
