from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import cv2
import tyro




@dataclass
class Args:
    src_images_dir: str
    annotations_path: str
    dst_images_dir: str
    dst_bboxes_dir: str


def get_box(annot):
    h = annot["original_height"]
    w = annot["original_width"]
    poly = np.array(annot["value"]["points"]) * np.array([w, h]) * 0.01

    x_min = int(np.min(poly[:, 0]))
    y_min = int(np.min(poly[:, 1]))
    x_max = int(np.max(poly[:, 0]))
    y_max = int(np.max(poly[:, 1]))

    box_cx = (x_min + x_max) / 2
    box_cy = (y_min + y_max) / 2
    box_w = x_max - x_min
    box_h = y_max - y_min

    label = annot["value"]["polygonlabels"][0]
    return {
        "x": box_cx / w,
        "y": box_cy / h,
        "w": box_w / w,
        "h": box_h / h,
        "label": label,
    }


def main():
    args = tyro.cli(Args)

    src_images_dir = Path(args.src_images_dir)
    annotations_path = Path(args.annotations_path)
    dst_images_dir = Path(args.dst_images_dir)
    dst_bboxes_dir = Path(args.dst_bboxes_dir)

    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_bboxes_dir.mkdir(parents=True, exist_ok=True)

    with open(annotations_path, "r") as f:
        data = json.load(f)

    counter = 0

    for item in data:
        image_name = Path(item["data"]["image"]).name.split("/")[-1]
        image_path = src_images_dir / image_name
        image = cv2.imread(str(image_path))
        boxes = []

        if len(item["annotations"]) != 1:
            print(f"Expected 1 annotation, got {len(item['annotations'])}")
            continue

        annot_results = item["annotations"][-1]["result"]
        if len(annot_results) == 0:
            print(f"No annotations found for {image_name}")
            continue

        for annot in annot_results:
            try:
                box = get_box(annot)
                boxes.append(box)
            except Exception as e:
                print(f"Error getting box for {image_name}: {e}")
                continue
          
        
        img_save_path = dst_images_dir / f"{counter:06d}.png"
        mask_bboxes_path = (dst_bboxes_dir / f"{counter:06d}.json")

        img_save_path.parent.mkdir(parents=True, exist_ok=True)
        mask_bboxes_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(img_save_path), image)
        with open(mask_bboxes_path, "w") as f:
            json.dump(boxes, f)

        counter += 1


if __name__ == "__main__":
    main()
