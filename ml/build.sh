#!/bin/bash

pip install torch torchvision 
pip install -r external/SAN/requirements.txt
pip install git+https://github.com/facebookresearch/detectron2.git@v0.6
pip install transformers pillow==9.5.0 fastapi uvicorn