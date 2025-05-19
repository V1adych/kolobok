from setuptools import setup, find_packages

setup(
    name="tyro",
    version="1.0",
    packages=find_packages("tyro", "tyro/*"),
    install_requires=[
        "numpy==1.22.4",
        "torch",
        "torchvision",
        "opencv-python",
        "open_clip_torch==2.16.0",
        "ftfy",
        "regex",
        "mmcv==1.3.14",
        "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@v0.6",
    ],
)
