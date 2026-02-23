import os
from rfdetr import RFDETRBase
from roboflow import download_dataset

os.environ["ROBOFLOW_API_KEY"] = "BRdDttL8wwHFrA27Xv07"


dataset = download_dataset("https://app.roboflow.com/koloboktyresegmentation/tire-spikes-det-gbicv/12", "coco")
model = RFDETRBase()
model.train(dataset_dir=dataset.location, epochs=15, batch_size=16, grad_accum_steps=1)


