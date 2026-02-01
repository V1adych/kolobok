import os
from rfdetr import RFDETRBase, RFDETRMedium
from roboflow import download_dataset

os.environ["ROBOFLOW_API_KEY"] = "BRdDttL8wwHFrA27Xv07"


dataset = download_dataset("https://app.roboflow.com/koloboktyresegmentation/tire-spikes-det-gbicv/12", "coco")
# model = RFDETRMedium()
# model.train(dataset.location, epochs=15, batch_size=16, grad_accum_steps=0)


