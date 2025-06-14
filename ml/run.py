import torch
import numpy as np
import cv2
from typing import List, Dict, Any
from torchvision.io import read_image
import argparse
import pprint

from tire_vision.thread.pipeline import TireVisionPipeline
from tire_vision.config import TireVisionConfig


def add_detection_boxes(image: np.ndarray, detections: List[Dict[str, Any]]):
    for detection in detections:
        x1, y1, w, h = detection["box"]
        class_name = detection["class"]
        cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
        cv2.putText(image, class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image

def main():
    parser = argparse.ArgumentParser(description="Tire Vision Pipeline")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    config = TireVisionConfig()
    pipeline = TireVisionPipeline(config)

    image = read_image(args.image_path)

    if image.shape[0] == 4:
        image = image[:3, :, :]

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    image = image

    result = pipeline(image)

    print("Analysis Result:")
    pprint.pprint(result)

    cropped_image = result["cropped_image"]
    cropped_image = add_detection_boxes(cropped_image, result["spikes"])

    cv2.imwrite("result.png", cropped_image)

if __name__ == "__main__":
    main()
