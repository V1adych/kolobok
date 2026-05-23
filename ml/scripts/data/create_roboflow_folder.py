from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
import tyro
from tqdm import tqdm

IMAGE_PATTERNS = ("**/*.jpg", "**/*.jpeg", "**/*.png")

@dataclass
class Args:
    input_dir: str
    output_dir: str
    prefix_name: str = ""



def collect_image_paths(input_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        image_paths.extend(sorted(input_dir.glob(pattern)))
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {input_dir}")
    return image_paths


def main():
    args = tyro.cli(Args)
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    image_paths = np.random.permutation(collect_image_paths(input_dir))

    pbar = tqdm(image_paths, desc="Copying images")
    for idx, image_path in enumerate(pbar):
        target_path = output_dir / f"{args.prefix_name}{idx:06d}.jpg"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path, target_path)


if __name__ == "__main__":
    main()
