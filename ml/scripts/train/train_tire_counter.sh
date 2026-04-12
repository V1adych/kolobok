#!/bin/bash

set -euo pipefail

python -c "
import os
from roboflow import download_dataset

os.environ['ROBOFLOW_API_KEY'] = 'BRdDttL8wwHFrA27Xv07'

dataset = download_dataset(
    'https://app.roboflow.com/koloboktyresegmentation/tire_count/2',
    'coco',
    location='data/tire_count'
)
"

python submodules/mmdetection/tools/train.py submodules/mmdetection/configs/rtmdet/tire_count.py --amp
