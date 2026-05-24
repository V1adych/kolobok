import onnxruntime
import numpy as np
import cv2

from tire_vision.config import DepthRegressorConfig, ort_providers, ort_opts


class DepthRegressor:
    def __init__(self, config: DepthRegressorConfig):
        self.config = config
        self.session = onnxruntime.InferenceSession(
            self.config.depth_regressor_onnx,
            providers=ort_providers,
            sess_options=ort_opts,
        )

    def resize(self, image: np.ndarray):
        return cv2.resize(image, self.config.resize_shape, interpolation=cv2.INTER_LINEAR)

    def forward(self, images: np.ndarray):
        return self.session.run(None, {"input": images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0})[0].squeeze(1)

    def __call__(self, image: np.ndarray):
        return self.forward(image)
