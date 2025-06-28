import base64
import json
from typing import Optional, Dict, Any
from PIL import Image
import numpy as np
import replicate
import io

from tire_vision.config import OCRConfig

import logging


class TireOCR:
    """OCR class for extracting tire information from images."""

    def __init__(self, config: OCRConfig):
        self.config = config

        self.logger = logging.getLogger("ocr")
        self.logger.info("TireOCR module initialized")

    def extract_tire_info(self, image: np.ndarray) -> Dict[str, Optional[str]]:
        """
        Extract tire information from an image.

        Args:
            image: Input image as numpy array (RGB format)

        Returns:
            Dictionary with manufacturer, model, and tire_size_string fields
        """

        pil_image = Image.fromarray(image)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)

        b64_data = base64.b64encode(buffer.read()).decode("utf-8")
        file_input = f"data:application/octet-stream;base64,{b64_data}"

        try:
            result = ""
            for event in replicate.stream(
                self.config.model_name,
                input={
                    "top_p": self.config.top_p,
                    "prompt": self.config.prompt,
                    "image_input": [file_input],
                    "temperature": self.config.temperature,
                    "presence_penalty": self.config.presence_penalty,
                    "frequency_penalty": self.config.frequency_penalty,
                    "max_completion_tokens": self.config.max_completion_tokens,
                },
            ):
                result += str(event)

            self.logger.info(f"OCR result: {result}")

            try:
                tire_info = json.loads(result.strip())
                self.logger.info(f"Parsed OCR result: {tire_info}")

                return {
                    "manufacturer": tire_info.get("manufacturer"),
                    "model": tire_info.get("model"),
                    "tire_size_string": tire_info.get("tire_size_string"),
                }

            except json.JSONDecodeError:
                self.logger.info(
                    f"JSONDecodeError: Failed to parse OCR result {result}"
                )
                self.logger.info(f"Falling back to default values")
                return {"manufacturer": None, "model": None, "tire_size_string": None}

        except Exception as e:
            self.logger.info(f"Error during OCR processing: {e}")
            self.logger.info(f"Falling back to default values")
            return {"manufacturer": None, "model": None, "tire_size_string": None}
