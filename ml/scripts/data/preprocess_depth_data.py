from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
import polars as pl
import tyro
from tqdm import tqdm

from tire_vision.config import ThreadSegmentatorConfig
from tire_vision.options import ThreadSegmentatorOptions
from tire_vision.thread.segmentator.pipeline import ThreadSegmentator, TireInstance

IMAGE_PATTERNS = ("*/*.jpg", "*/*.jpeg", "*/*.png")
TARGET_RE = re.compile(r"^(\d+(?:,\d+)?)")


@dataclass
class Args:
    input_dir: str
    output_dir: str
    output_shape: tuple[int, int] = (640, 640)
    segmentator_config: ThreadSegmentatorConfig = ThreadSegmentatorConfig()
    confidence_threshold: float = 0.5
    nms_iou_threshold: float = 0.2
    pre_topk: int = 200
    mask_threshold: float = 0.5
    padding_frac: float = 0.01
    min_tire_pixels: int = 96


def parse_target(path: Path) -> float:
    match = TARGET_RE.match(path.parent.name)
    if match is None:
        raise ValueError(f"Could not parse target from directory name: {path.parent.name}")
    return float(match.group(1).replace(",", "."))


def collect_image_paths(input_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        image_paths.extend(sorted(input_dir.glob(pattern)))
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {input_dir}")
    return image_paths


def load_rgb_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_tires(image: np.ndarray, tires: list[TireInstance]) -> np.ndarray:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    tire_colors = [(255, 0, 0), (0, 180, 255), (255, 0, 255), (0, 255, 0)]

    for idx, tire in enumerate(tires):
        x, y, w, h = tire.box
        tire_color = tire_colors[idx % len(tire_colors)]
        mask = tire.mask.astype(bool)
        image_bgr[mask] = (0.6 * image_bgr[mask] + 0.4 * np.array(tire_color, dtype=np.uint8)).astype(np.uint8)
        cv2.rectangle(image_bgr, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), tire_color, 3)
        cv2.putText(
            image_bgr,
            f"tire {tire.score:.2f}",
            (x - w // 2, max(0, y - h // 2 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            tire_color,
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_existing_rows(labels_path: Path) -> tuple[dict[str, list], int, int]:
    if not labels_path.exists():
        return {"vis_name": [], "image_name": [], "label": []}, 0, 0

    existing = pl.read_csv(labels_path)
    vis_names = existing["vis_name"].to_list()
    image_names = existing["image_name"].to_list()
    labels = existing["label"].to_list()

    next_image_index = max(int(Path(image_name).stem) for image_name in image_names) + 1 if image_names else 0
    next_vis_index = max(int(Path(vis_name).stem) for vis_name in vis_names) + 1 if vis_names else 0
    return {"vis_name": vis_names, "image_name": image_names, "label": labels}, next_image_index, next_vis_index


def main():
    args = tyro.cli(Args)
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    processed_dir = output_dir / "images"
    vis_dir = output_dir / "vis"
    labels_path = output_dir / "labels.csv"
    processed_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    segmentator_options = ThreadSegmentatorOptions(
        confidence_threshold=args.confidence_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        pre_topk=args.pre_topk,
        mask_threshold=args.mask_threshold,
        padding_frac=args.padding_frac,
        min_tire_pixels=args.min_tire_pixels,
    )

    image_paths = collect_image_paths(input_dir)
    parsed_inputs = [(path, parse_target(path)) for path in image_paths]

    segmentator = ThreadSegmentator(args.segmentator_config)
    rows, next_index, next_vis_index = load_existing_rows(labels_path)

    pbar = tqdm(parsed_inputs, desc="Processing images")
    for image_path, target in pbar:
        image = load_rgb_image(image_path)
        tires = segmentator(image, options=segmentator_options)
        if len(tires) == 0:
            raise ValueError(f"No tires detected for image: {image_path}")
        pbar.set_postfix(num_tires=len(tires))

        vis_name = f"{next_vis_index:06d}.png"
        vis_path = vis_dir / vis_name
        vis = draw_tires(image, tires)
        if not cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)):
            raise ValueError(f"Failed to write visualization: {vis_path}")
        next_vis_index += 1

        for tire in tires:
            cropped = segmentator.crop_tire(image, tire, options=segmentator_options)
            if cropped is None:
                raise ValueError(f"Failed to crop tire for image: {image_path}")

            resized = cv2.resize(cropped, args.output_shape, interpolation=cv2.INTER_LINEAR)
            image_name = f"{next_index:06d}.png"
            save_path = processed_dir / image_name
            if not cv2.imwrite(str(save_path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)):
                raise ValueError(f"Failed to write image: {save_path}")

            rows["vis_name"].append(vis_name)
            rows["image_name"].append(image_name)
            rows["label"].append(target)
            next_index += 1

    pl.DataFrame(rows).write_csv(labels_path)


if __name__ == "__main__":
    main()
