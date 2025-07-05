import base64
import json
from typing import Optional, Dict, Any, List
from PIL import Image
import numpy as np
from openai import OpenAI
import io
from traceback import format_exc
import logging
import re

from tire_vision.config import OCRConfig


class TireOCR:
    def __init__(self, config: OCRConfig):
        self.config = config
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

        self.logger = logging.getLogger("ocr")
        self.logger.info("TireOCR module initialized")

    def extract_tire_info(
        self,
        images: list[np.ndarray],
        prompt: str,
    ) -> Dict[str, list[str]]:
        file_inputs = [self._prepare_image(img) for img in images]

        try:
            result = self._get_llm_response(file_inputs, prompt)
            tire_info = self._parse_llm_response(result)
            return tire_info
        except Exception:
            self.logger.error(format_exc())
            self.logger.error(
                "Error during OCR processing. Falling back to default values"
            )
            return self._get_default_response()

    def _prepare_image(self, image: np.ndarray) -> str:
        pil_image = Image.fromarray(image)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)

        b64_data = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_data}"

    def _get_llm_response(self, file_inputs: list[str], prompt: str) -> str:
        result = ""
        content = [
            {"type": "text", "text": prompt},
        ]
        for file_input in file_inputs:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": file_input},
                }
            )
        
        stream = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            stream=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_completion_tokens,
            presence_penalty=self.config.presence_penalty,
            frequency_penalty=self.config.frequency_penalty,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                result += chunk.choices[0].delta.content

        self.logger.info(f"LLM response: {result}")
        return result

    def _parse_llm_response(self, result: str) -> Dict[str, list[str]]:
        match = re.search(r"\{.*\}", result, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response")
        json_str = match.group(0)
        tire_info = json.loads(json_str)
        self.logger.info(f"Parsed OCR result: {tire_info}")

        return {
            "strings": tire_info.get("strings", []),
        }

    def _get_default_response(self) -> Dict[str, list[str]]:
        self.logger.info("Falling back to default values")
        return {"strings": []}
