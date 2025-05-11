from pathlib import Path

import sys
import numpy as np
import cv2
import torch
from shapely.geometry import Polygon
from matplotlib import pyplot as plt

from inference import get_model
model = get_model("tire-segmentation-eqoeu/5", api_key="BRdDttL8wwHFrA27Xv07")

image = cv2.imread(Path(sys.argv[1]))

result = model.infer(image)

pts1 = np.array([[p.x, p.y] for p in result[0].predictions[1].points]).astype(np.int32)
pts2 = np.array([[p.x, p.y] for p in result[0].predictions[0].points]).astype(np.int32)

mask = cv2.fillPoly(
    cv2.fillPoly(
        np.zeros_like(image, dtype=np.uint8),
        [pts1],
        (255, 255, 255),
        lineType=cv2.LINE_AA,
    ),
    [pts2],
    (0, 0, 0),
    lineType=cv2.LINE_AA,
).astype(np.uint8)

# get bounding box
poly = Polygon(pts1)
x, y = poly.exterior.xy
x = np.array(x).astype(np.int32)
y = np.array(y).astype(np.int32)
x_min = np.min(x)
x_max = np.max(x)
y_min = np.min(y)
y_max = np.max(y)

bb = np.array([
    [x_min, y_min],
    [x_max, y_min],
    [x_max, y_max],
    [x_min, y_max],
]).astype(np.float32)

target_size = (x_max - x_min, y_max - y_min)

dst_bb = np.array([
    [10, 10],
    [target_size[0] - 10, 10],
    [target_size[0] - 10, target_size[1] - 10],
    [10, target_size[1] - 10],
]).astype(np.float32)

transform = cv2.getPerspectiveTransform(bb, dst_bb)
warped_img = cv2.warpPerspective(image, transform, target_size)
warped_mask = cv2.warpPerspective(mask, transform, target_size)

xc, yc, _ = warped_img.shape
yc //= 2
xc //= 2
center = xc, yc

output_size = (xc, int(2 * np.pi * xc))

strip = cv2.warpPolar(
    warped_img,
    output_size,
    center=center,
    maxRadius=xc * 1.1,
    flags=cv2.INTER_CUBIC,
)
strip_mask = cv2.warpPolar(
    warped_mask,
    output_size,
    center=center,
    maxRadius=xc * 1.1,
    flags=cv2.INTER_CUBIC,
)

cut = strip[:, np.mean(strip_mask[:, :, 0] / 255, axis=0) > 0.5].transpose(1, 0, 2)[::-1]

clahe = cv2.createCLAHE(clipLimit=5)

lab = cv2.cvtColor(cut, cv2.COLOR_BGR2LAB)
lab_planes = list(cv2.split(lab))
lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

cv2.imwrite(Path(sys.argv[2]), img_clahe)