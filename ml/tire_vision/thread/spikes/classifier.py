import onnxruntime
import numpy as np

from tire_vision.config import ort_providers, ort_opts


class SpikeClassifier:
    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self.session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=ort_providers,
            sess_options=ort_opts,
        )

    def forward(self, image: np.ndarray):
        return self.session.run(None, {"input": image})[0]

    def __call__(self, image: np.ndarray):
        return self.forward(image)
