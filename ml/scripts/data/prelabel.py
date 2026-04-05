from dataclasses import dataclass
from pathlib import Path
import sys
import json
import hashlib

sys.path.append(str(Path(__file__).parent.parent.parent))

import tyro
import cv2
from tqdm import tqdm
from tire_vision.thread.studs.pipeline import StudPipeline, StudPipelineConfig, StudPipelineOptions


@dataclass
class Args:
    src_dir: str
    dst_dir: str
    onnx_path: str
    prefix_name: str
    dry_run: bool = False
    # If > 0, split outputs into dst_dir/part0, part1, ... each with <= batch_size images + its own annotations.json
    batch_size: int = 0
    confidence_threshold: float = 0.25


categories = [
    {"id": 0, "name": "spikes", "supercategory": "none"},
    {"id": 1, "name": "absent", "supercategory": "spikes"},
    {"id": 2, "name": "broken", "supercategory": "spikes"},
    {"id": 3, "name": "floating", "supercategory": "spikes"},
    {"id": 4, "name": "normal", "supercategory": "spikes"},
    {"id": 5, "name": "unsure", "supercategory": "spikes"},
]

category_mapping = {
    "absent": 1,
    "healthy": 4,
    "broken": 2,
    "floating": 3,
    "indistinguishable": 5,
}


def get_images(src_dir: Path) -> list[Path]:
    extensions = ["jpg", "jpeg", "png"]
    images: list[Path] = []
    for ext in extensions:
        images.extend(src_dir.glob(f"**/*.{ext}"))
        images.extend(src_dir.glob(f"**/*.{ext.upper()}"))
    return images


def _new_coco():
    return {
        "images": [],
        "annotations": [],
        "categories": categories,
    }


def _part_dir(dst_dir: Path, batch_size: int, part_idx: int) -> Path:
    # If batch_size <= 0, keep everything directly in dst_dir
    return (dst_dir / f"part{part_idx}") if batch_size > 0 else dst_dir


def _finalize_part(part_dir: Path, coco_annot: dict) -> None:
    part_dir.mkdir(parents=True, exist_ok=True)
    with open(part_dir / "annotations.json", "w") as f:
        json.dump(coco_annot, f)


def _hash_image_pixels(image) -> bytes:
    h = hashlib.sha256()
    h.update(str(image.shape).encode("utf-8"))
    h.update(str(image.dtype).encode("utf-8"))
    h.update(image.tobytes())
    return h.digest()


def main():
    args = tyro.cli(Args)
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)

    if args.batch_size < 0:
        raise ValueError("batch_size must be >= 0")

    cfg = StudPipelineConfig(spike_detector_onnx=args.onnx_path)
    pipe = StudPipeline(cfg)

    images = get_images(src_dir)
    if args.dry_run:
        images = images[:10]

    total_images = len(images)
    msg = f"Will label {total_images} images"
    if args.dry_run:
        msg += " [DRY RUN]"
    if args.batch_size > 0:
        msg += f" (batch_size={args.batch_size})"
    print(msg)

    pbar = tqdm(total=total_images, desc="Labeling images")

    part_idx = 0
    coco_annot = _new_coco()
    num_annotations = 0
    seen_hashes: set[bytes] = set()
    skipped_duplicates = 0
    read_failed = 0
    saved_images = 0

    for global_i, image_path in enumerate(images):
        # determine which part we are in
        if args.batch_size > 0:
            new_part_idx = global_i // args.batch_size
            if new_part_idx != part_idx:
                # finalize previous part
                prev_part_dir = _part_dir(dst_dir, args.batch_size, part_idx)
                if not args.dry_run:
                    _finalize_part(prev_part_dir, coco_annot)

                # start new part
                part_idx = new_part_idx
                coco_annot = _new_coco()
                num_annotations = 0

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR_RGB)
        if image is None:
            read_failed += 1
            pbar.set_postfix(read_failed=read_failed, dup=skipped_duplicates, saved=saved_images)
            pbar.update(1)
            continue

        image_hash = _hash_image_pixels(image)
        if image_hash in seen_hashes:
            skipped_duplicates += 1
            pbar.set_postfix(dup=skipped_duplicates, saved=saved_images)
            pbar.update(1)
            continue
        seen_hashes.add(image_hash)

        studs, _, _ = pipe(image, options=StudPipelineOptions(confidence_threshold=args.confidence_threshold))
        saved_images += 1
        pbar.set_postfix(num_studs=len(studs), dup=skipped_duplicates, saved=saved_images)

        part_dir = _part_dir(dst_dir, args.batch_size, part_idx)

        # local index within part is used as image_id (and for naming)
        local_i = global_i if args.batch_size <= 0 else (global_i % args.batch_size)

        if not args.dry_run:
            save_path = part_dir / f"{args.prefix_name}_{local_i}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            coco_annot["images"].append(
                {
                    "id": local_i,
                    "file_name": str(save_path.relative_to(part_dir)),
                    "width": image.shape[1],
                    "height": image.shape[0],
                }
            )
            for stud in studs:
                category_id = category_mapping[stud.label]
                xc, yc, w, h = stud.box
                x_min = int(round(xc - w / 2))
                y_min = int(round(yc - h / 2))
                coco_annot["annotations"].append(
                    {
                        "id": num_annotations,
                        "image_id": local_i,
                        "category_id": category_id,
                        "bbox": [x_min, y_min, w, h],
                        "area": int(w * h),
                        "segmentation": [],
                        "iscrowd": 0,
                    }
                )
                num_annotations += 1

            cv2.imwrite(str(save_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        pbar.update(1)

    pbar.close()

    # finalize last part
    if not args.dry_run:
        last_part_dir = _part_dir(dst_dir, args.batch_size, part_idx)
        _finalize_part(last_part_dir, coco_annot)

    print(f"{saved_images}/{total_images}")


if __name__ == "__main__":
    main()
