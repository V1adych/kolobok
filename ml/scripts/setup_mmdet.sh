#!/bin/bash

set -euo pipefail

pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install mmengine tensorboard tensorboardX
pip install -r submodules/mmdetection/requirements/runtime.txt
pip install albumentations roboflow
pip install -e submodules/mmdetection
