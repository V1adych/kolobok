import base64
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import replace, dataclass
from PIL import Image
import io
from traceback import format_exc
import re

import numpy as np
from openai import OpenAI, APIStatusError
from fastapi import HTTPException
from tire_vision.config import OCRConfig
from tire_vision.options import OCROptions

import logging


@dataclass
class OCRResult:
    strings: List[str]
    tire_size: str


class OCRPipeline:
    def __init__(self, config: OCRConfig):
        self.config = config
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

        self.examples_dir = Path(__file__).parent / "examples"

        self.logger = logging.getLogger("ocr")
        self.logger.info("TireOCR module initialized")

    def extract_tire_info(
        self,
        images: List[np.ndarray],
        prompt: str,
        options: Optional[OCROptions] = None,
    ) -> OCRResult:
        file_inputs = [self._prepare_image(img) for img in images]
        if options is not None:
            self.config = replace(self.config, options=options)

        user_prompt = self._build_user_prompt(prompt)
        try:
            response_text = self._get_llm_response(file_inputs, user_prompt)
        except APIStatusError as e:
            self.logger.error(format_exc())
            raise HTTPException(
                status_code=e.status_code,
                detail=f"OCR provider returned error: {e.body}",
            )

        tire_info = self._parse_llm_response(response_text)
        return tire_info

    def __call__(
        self,
        images: List[np.ndarray],
        prompt: str,
        options: Optional[OCROptions] = None,
    ) -> Dict[str, List[str]]:
        return self.extract_tire_info(images, prompt, options=options)

    def _build_user_prompt(self, prompt: str) -> str:
        base_prompt = self.config.prompt
        return f"{base_prompt} {prompt}"

    def _prepare_image(self, image: np.ndarray) -> str:
        pil_image = Image.fromarray(image)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)

        b64_data = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_data}"

    def _build_messages(self, file_inputs: list[str], user_prompt: str) -> list[dict]:
        messages = []

        messages.append({"role": "system", "content": self.config.system_prompt})

        content = [
            {"type": "text", "text": user_prompt},
        ]
        for file_input in file_inputs:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": file_input},
                }
            )

        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

        return messages

    def _get_request_kwargs(self, messages: List[dict]) -> Dict[str, Any]:
        params = dict(
            model=self.config.options.model_name,
            messages=messages,
            stream=False,
            temperature=self.config.options.temperature,
            top_p=self.config.options.top_p,
            max_tokens=self.config.options.max_completion_tokens,
            presence_penalty=self.config.options.presence_penalty,
            frequency_penalty=self.config.options.frequency_penalty,
        )

        if self.config.options.providers_list:
            params["extra_body"] = {"provider": {"only": self.config.options.providers_list}}

        return params

    def _get_llm_response(self, file_inputs: List[str], user_prompt: str) -> Optional[str]:
        messages = self._build_messages(file_inputs, user_prompt)

        response = self.client.chat.completions.create(**self._get_request_kwargs(messages))
        response_text = response.choices[0].message.content
        self.logger.info(f"OCR response: {response_text}")

        return response_text

    def _parse_llm_response(self, response: str) -> OCRResult:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        error_detail = f"OCR response is not valid: {response}"
        if not match:
            raise HTTPException(status_code=502, detail=error_detail)
        json_str = match.group(0)
        tire_info = json.loads(json_str)
        self.logger.info(f"Parsed OCR result: {tire_info}")

        if "strings" not in tire_info or "tire_size" not in tire_info:
            raise HTTPException(status_code=502, detail=error_detail)

        return OCRResult(strings=tire_info["strings"], tire_size=tire_info["tire_size"])
