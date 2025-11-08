from pathlib import Path
import shutil
import sys
from dataclasses import dataclass

import tyro

from sklearn.model_selection import train_test_split


sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class Args:
    src_dir: str
    train_dir: str
    val_dir: str
    images_name: str = "images"
    annot_name: str = "annot"
    test_size: float = 0.2


def main():
    args = tyro.cli(Args)

    pairs = []

    src_images = Path(args.src_dir) / args.images_name
    src_annot = Path(args.src_dir) / args.annot_name

    dest_images_train = Path(args.train_dir) / args.images_name
    dest_annot_train = Path(args.train_dir) / args.annot_name

    dest_images_val = Path(args.val_dir) / args.images_name
    dest_annot_val = Path(args.val_dir) / args.annot_name

    dest_images_train.mkdir(parents=True, exist_ok=True)
    dest_annot_train.mkdir(parents=True, exist_ok=True)
    dest_images_val.mkdir(parents=True, exist_ok=True)
    dest_annot_val.mkdir(parents=True, exist_ok=True)

    for image_path in src_images.iterdir():
        annot_path = src_annot / image_path.with_suffix(".json").name

        if not annot_path.exists():
            print(f"Annot not found for {image_path}")
            continue

        pairs.append((image_path, annot_path))

    train_pairs, val_pairs = train_test_split(pairs, test_size=args.test_size, random_state=42)

    for image_path, annot_path in train_pairs:
        shutil.copy(image_path, dest_images_train / image_path.name)
        shutil.copy(annot_path, dest_annot_train / annot_path.name)

    for image_path, annot_path in val_pairs:
        shutil.copy(image_path, dest_images_val / image_path.name)
        shutil.copy(annot_path, dest_annot_val / annot_path.name)


if __name__ == "__main__":
    main()
