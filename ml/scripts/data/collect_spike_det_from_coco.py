from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import json

import cv2
import tyro
import urllib.parse


def find_longest_common_prefix(s1: str, s2: str) -> int:
    """Return length of the longest common prefix between two strings."""
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return i
    return min_len


def find_best_match(target_path: Path, images_dir: Path) -> Path | None:
    """Find the image file with the longest common prefix with the target path name.

    Only considers files with extensions: .jpg, .jpeg, .png
    """
    target_name = target_path.name
    valid_extensions = {".jpg", ".jpeg", ".png"}

    image_files: List[Path] = [
        img_file
        for img_file in images_dir.iterdir()
        if img_file.is_file() and img_file.suffix.lower() in valid_extensions
    ]

    if not image_files:
        return None

    prefix_lengths = [
        find_longest_common_prefix(target_name, img.name) for img in image_files
    ]
    max_idx = max(range(len(prefix_lengths)), key=prefix_lengths.__getitem__)
    return image_files[max_idx] if prefix_lengths[max_idx] > 0 else None


def normalized_yolo_bbox(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> Dict[str, float]:
    cx = (x + w / 2.0) / float(img_w)
    cy = (y + h / 2.0) / float(img_h)
    nw = w / float(img_w)
    nh = h / float(img_h)
    return {"x": cx, "y": cy, "w": nw, "h": nh}


@dataclass
class Args:
    input_images_path: str
    annotations_path: str
    dst_images: str
    dst_annot: str


def main():
    args = tyro.cli(Args)

    src_images_dir = Path(args.input_images_path)
    annotations_path = Path(args.annotations_path)
    dst_images_dir = Path(args.dst_images)
    dst_bboxes_dir = Path(args.dst_annot)

    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_bboxes_dir.mkdir(parents=True, exist_ok=True)

    with open(annotations_path, "r") as f:
        data = json.load(f)

    images: List[Dict] = data.get("images", [])
    annots: List[Dict] = data.get("annotations", [])
    categories: List[Dict] = data.get("categories", [])

    # Build category id -> name mapping
    cat_id_to_name: Dict[int, str] = {
        c.get("id"): c.get("name", str(c.get("id"))) for c in categories
    }

    # Build image_id -> list[annotation] index
    img_id_to_annots: Dict[int, List[Dict]] = {}
    for a in annots:
        img_id = a.get("image_id")
        if img_id is None:
            continue
        img_id_to_annots.setdefault(img_id, []).append(a)

    counter = 0

    for item in images:
        # Resolve image path from COCO. Prefer "path", fallback to "file_name".
        raw_path = item.get("path") or item.get("file_name") or ""
        img_name_candidate = Path(raw_path).name

        # Handle exporter that prefixes with an id and a dash; try to strip the prefix as in notebook
        if "-" in img_name_candidate:
            try:
                img_name_candidate = img_name_candidate.split("-", maxsplit=1)[1]
            except Exception:
                pass

        img_name_candidate = urllib.parse.unquote(img_name_candidate)

        candidate_path = src_images_dir / img_name_candidate
        if not candidate_path.exists():
            best_match = find_best_match(candidate_path, src_images_dir)
            if best_match is None:
                print(f"Image not found for '{img_name_candidate}', skipping.")
                continue
            candidate_path = best_match

        img_w = item.get("width")
        img_h = item.get("height")
        if not img_w or not img_h:
            # Fallback to reading image to get shape
            img = cv2.imread(str(candidate_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to read image: {candidate_path}")
                continue
            img_h, img_w = img.shape[:2]
        else:
            img = cv2.imread(str(candidate_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to read image: {candidate_path}")
                continue

        # Collect boxes for this image
        image_id = item.get("id")
        relevant_annots = img_id_to_annots.get(image_id, [])
        if len(relevant_annots) == 0:
            # No annotations; skip
            print(
                f"No annotations for image_id={image_id} ({candidate_path.name}), skipping."
            )
            continue

        boxes: List[Dict] = []
        for annot in relevant_annots:
            bbox = annot.get("bbox")  # [x, y, w, h]
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            norm = normalized_yolo_bbox(x, y, w, h, img_w, img_h)
            cat_id = annot.get("category_id")
            label = cat_id_to_name.get(
                cat_id, str(cat_id) if cat_id is not None else ""
            )
            boxes.append(
                {
                    "x": norm["x"],
                    "y": norm["y"],
                    "w": norm["w"],
                    "h": norm["h"],
                    "label": label,
                }
            )

        if len(boxes) == 0:
            # No valid boxes; skip
            print(
                f"No valid boxes for image_id={image_id} ({candidate_path.name}), skipping."
            )
            continue

        # Save sequentially to align with the LS converter convention
        img_save_path = dst_images_dir / f"{counter:06d}.png"
        boxes_save_path = dst_bboxes_dir / f"{counter:06d}.json"

        img_save_path.parent.mkdir(parents=True, exist_ok=True)
        boxes_save_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(img_save_path), img)
        with open(boxes_save_path, "w") as f:
            json.dump(boxes, f)

        counter += 1


if __name__ == "__main__":
    main()
