from pathlib import Path
import shutil
import sys
from dataclasses import dataclass

import tyro
import pandas as pd

from sklearn.model_selection import train_test_split


sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class Args:
    src_dir: str
    train_dir: str
    val_dir: str
    df_path: str
    test_size: float = 0.2


def main():
    args = tyro.cli(Args)

    src_images = Path(args.src_dir)
    src_df = pd.read_csv(args.df_path)
    dest_images_train = Path(args.train_dir) / "images"
    dest_df_train = Path(args.train_dir) / "labels.csv"
    dest_images_val = Path(args.val_dir) / "images"
    dest_df_val = Path(args.val_dir) / "labels.csv"

    dest_images_train.mkdir(parents=True, exist_ok=True)
    dest_images_val.mkdir(parents=True, exist_ok=True)

    train_df, val_df = train_test_split(src_df, test_size=args.test_size, random_state=42)

    train_df.to_csv(dest_df_train, index=False)
    val_df.to_csv(dest_df_val, index=False)

    for _, row in train_df.iterrows():
        image_name = row["image_name"]
        src_image_path = src_images / image_name
        dest_image_path = dest_images_train / image_name
        dest_image_path.parent.mkdir(parents=True, exist_ok=True)
        if not src_image_path.exists():
            print(f"Image {src_image_path} does not exist")
            continue
        shutil.copy(src_image_path, dest_image_path)

    for _, row in val_df.iterrows():
        image_name = row["image_name"]
        src_image_path = src_images / image_name
        dest_image_path = dest_images_val / image_name
        dest_image_path.parent.mkdir(parents=True, exist_ok=True)
        if not src_image_path.exists():
            print(f"Image {src_image_path} does not exist")
            continue
        shutil.copy(src_image_path, dest_image_path)


if __name__ == "__main__":
    main()
