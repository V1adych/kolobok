import numpy as np
import onnxruntime as ort

from tire_vision.config import ThreadSegmentatorConfig, ort_opts, ort_providers


class ThreadSegmentatorModel:
    def __init__(self, config: ThreadSegmentatorConfig):
        self.config = config
        self.detector_session = ort.InferenceSession(config.detector_onnx, providers=ort_providers, sess_options=ort_opts)
        self.mask_decoder_session = ort.InferenceSession(config.mask_decoder_onnx, providers=ort_providers, sess_options=ort_opts)

    def detect(self, image_input: np.ndarray):
        return self.detector_session.run(
            ["boxes", "scores", "kernels", "priors", "mask_feat"],
            {"input": image_input},
        )

    def decode_masks(
        self,
        mask_feat: np.ndarray,
        kernels: np.ndarray,
        priors: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        return self.mask_decoder_session.run(
            ["mask_logits"],
            {
                "mask_feat": mask_feat.astype(np.float32),
                "kernels": kernels.astype(np.float32),
                "priors": priors.astype(np.float32),
                "valid": valid.astype(np.float32),
            },
        )[0]
