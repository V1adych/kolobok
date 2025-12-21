import albumentations as A
from ultralytics import YOLO, RTDETR
import cv2

DATA_YAML = "/root/workspace/tire-spikes-det-11/data.yaml"

augs = [
    A.Affine(
        scale=(0.75, 1.25),
        translate_percent=(-0.15, 0.15),
        rotate=(-25, 25),
        shear=(-12, 12),
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        p=0.7,
    ),
    # A.OneOf(
    #     [
    #         A.Perspective(scale=(0.02, 0.08), keep_size=True, border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
    #         A.GridDistortion(num_steps=5, distort_limit=0.20, p=1.0),
    #     ],
    #     p=0.25,
    # ),
    # A.OneOf(
    #     [
    #         A.GaussNoise(std_range=(0.1, 0.2), per_channel=True, p=1.0),
    #         A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.10, 0.20), p=1.0),
    #         A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1.0),
    #     ],
    #     p=0.5,
    # ),
    A.OneOf(
        [
            A.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.1, p=1.0),
            A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=45, val_shift_limit=30, p=1.0),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
            A.RandomToneCurve(scale=0.5, p=1.0),
            A.CLAHE(clip_limit=(1.0, 6.0), tile_grid_size=(8, 8), p=1.0),
        ],
        p=0.5,
    ),
    # A.OneOf(
    #     [
    #         A.Equalize(p=1.0),
    #         A.Posterize(num_bits=(3, 6), p=1.0),
    #         A.Solarize(threshold_range=(0.3, 0.7), p=1.0),
    #     ],
    #     p=0.5,
    # ),
    # A.ImageCompression(quality_range=(90, 100), p=0.3),
    A.Downscale(scale_range=(0.5, 0.9), p=0.2),
    # A.CoarseDropout(
    #     num_holes_range=(1, 3),
    #     hole_height_range=(0.05, 0.1),
    #     hole_width_range=(0.05, 0.1),
    #     fill=0,
    #     p=0.3,
    # ),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ChannelDropout(p=0.2),
    A.ChannelShuffle(p=0.2),
]

model = YOLO("yolo12m.pt")
model.train(
    project="runs/yolo12",
    name="yolo12",
    data=DATA_YAML,
    imgsz=640,
    epochs=100,
    batch=16,
    nbs=32,
    device=0,
    workers=4,
    cache=True,
    optimizer="AdamW",
    lr0=1e-4,
    augmentations=augs,
    mosaic=1.0,
    mixup=0.10,
    cutmix=0.0,
    copy_paste=0.0,
    close_mosaic=10,
)
