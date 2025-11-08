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

    def forward(self, image: np.ndarray):
        resized_image = cv2.resize(image, self.config.resize_shape, interpolation=cv2.INTER_LINEAR)
        image_input = resized_image.transpose(2, 0, 1)[None].astype(np.float32) / 255

        result = self.session.run(None, {"input": image_input})[0]
        result_scaled = 10 / (1 + np.exp(-np.squeeze(result)))

        return float(result_scaled)

    def __call__(self, image: np.ndarray):
        return self.forward(image)
