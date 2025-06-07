import json
from pathlib import Path

import numpy as np
import cv2

from matplotlib import pyplot as plt, colormaps as cm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from torchvision.transforms import functional as VF, InterpolationMode

from tqdm import tqdm
from torch import nn
from tire_vision.spike_counter.model import get_model

from mmengine.registry import MODELS

images_dir = Path("data/dataset_crop")
labels_path = Path("data/result.json")

# model = get_model()
# class Module(nn.Module):
#     def __init__(self, *args, **kwargs):
#         print("="*100)
#         print(*args, **kwargs)
# MODELS.register_module("PermuteLayerNorm", Module)
print(MODELS)