from typing import Tuple, Optional

import numpy as np
import cv2
import onnxruntime as ort

from tire_vision.config import ort_providers, ort_opts


class OnnxSegmentator:
    def __init__(
        self,
        onnx_path: str,
        resize_shape: Tuple[int, int],
        resize_mask_shape: Optional[Tuple[int, int]] = None,
    ):
        self.onnx_path = onnx_path
        self.resize_shape = resize_shape
        self.resize_mask_shape = resize_mask_shape
        self.session = ort.InferenceSession(
            self.onnx_path,
            providers=ort_providers,
            sess_options=ort_opts,
        )

    def forward(self, image: np.ndarray, threshold: Optional[float] = None):
        """
        Args:
            image: np.ndarray (H, W, 3), np.uint8,

        Returns:
            mask: np.ndarray (H, W), np.uint8 if threshold is None else np.float32 (normalized to [0, 1]),
        """
        h, w, _ = image.shape

        resized_image = (
            cv2.resize(image, self.resize_shape, interpolation=cv2.INTER_LINEAR)
            .transpose(2, 0, 1)[None]
            .astype(np.float32)
            / 255
        )

        logits = self.session.run(
            None,
            {"input": resized_image},
        )[0]

        logits_squeezed = np.squeeze(logits, axis=(0, 1))
        probs = 1 / (1 + np.exp(-logits_squeezed))
        resize_mask_shape = (w, h) if self.resize_mask_shape is None else self.resize_mask_shape
        probs = cv2.resize(probs, resize_mask_shape, interpolation=cv2.INTER_LINEAR)

        if threshold is not None:
            return (probs > threshold).astype(np.uint8) * 255

        return probs

    def __call__(self, image: np.ndarray, threshold: Optional[float] = None):
        return self.forward(image, threshold=threshold)
