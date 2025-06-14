import numpy as np
import cv2 
import torch

from tire_vision.thread.pipeline import TireVisionPipeline
from tire_vision.config import TireVisionConfig


cfg = TireVisionConfig()
pipeline = TireVisionPipeline(cfg)


def get_thread_stats(image: np.ndarray) -> dict:
    image = torch.from_numpy(image).permute(2, 0, 1)
    result = pipeline(image)
    return result