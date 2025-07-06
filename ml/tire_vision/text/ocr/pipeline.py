import base64
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image
import io
from traceback import format_exc
import re

import numpy as np
import cv2
from openai import OpenAI, AsyncOpenAI
from tire_vision.config import OCRConfig

import logging


class OCRPipeline:
    def __init__(self, config: OCRConfig):
        self.config = config
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
        self.async_client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

        self.examples_dir = Path(__file__).parent / "examples"

        self.logger = logging.getLogger("ocr")
        self.logger.info("TireOCR module initialized")

        self.few_shot_examples = []
        # self.few_shot_examples = self._load_few_shot_examples()

    def _load_few_shot_examples(self) -> List[Dict[str, Any]]:
        examples = []

        if not self.examples_dir.exists():
            self.logger.warning(f"Examples directory not found: {self.examples_dir}")
            return examples

        prompt_files = sorted(self.examples_dir.glob("prompt*.txt"))

        for prompt_file in prompt_files:
            try:
                number = prompt_file.stem.replace("prompt", "")
                answer_file = self.examples_dir / f"answer{number}.json"
                original_image_file = self.examples_dir / f"original{number}.png"
                unwrapped_image_file = self.examples_dir / f"unwrapped{number}.png"

                if not answer_file.exists() or not original_image_file.exists():
                    continue

                prompt_text = prompt_file.read_text().strip()
                answer_text = answer_file.read_text().strip()

                example_images = []

                original_img = cv2.imread(str(original_image_file))
                if original_img is not None:
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    example_images.append(self._prepare_image(original_img))

                # if unwrapped_image_file.exists():
                #     unwrapped_img = cv2.imread(str(unwrapped_image_file))
                #     if unwrapped_img is not None:
                #         unwrapped_img = cv2.cvtColor(unwrapped_img, cv2.COLOR_BGR2RGB)
                #         example_images.append(self._prepare_image(unwrapped_img))

                if example_images:
                    examples.append(
                        {
                            "user": prompt_text,
                            "assistant": answer_text,
                            "images": example_images,
                        }
                    )

            except Exception as e:
                self.logger.warning(f"Failed to load example from {prompt_file}: {e}")

        self.logger.info(f"Loaded {len(examples)} few-shot examples")
        return examples

    def extract_tire_info(
        self,
        images: list[np.ndarray],
    ) -> Dict[str, list[str]]:
        file_inputs = [self._prepare_image(img) for img in images]

        try:
            user_prompt = self._build_user_prompt(len(images))
            result = self._get_llm_response(file_inputs, user_prompt)
            tire_info = self._parse_llm_response(result)
            return tire_info
        except Exception:
            self.logger.error(format_exc())
            self.logger.error(
                "Error during OCR processing. Falling back to default values"
            )
            return self._get_default_response()

    async def async_extract_tire_info(
        self,
        images: list[np.ndarray],
    ) -> Dict[str, list[str]]:
        file_inputs = [self._prepare_image(img) for img in images]

        try:
            user_prompt = self._build_user_prompt(len(images))
            result = await self._async_get_llm_response(file_inputs, user_prompt)
            tire_info = self._parse_llm_response(result)
            return tire_info
        except Exception:
            self.logger.error(format_exc())
            self.logger.error(
                "Error during OCR processing. Falling back to default values"
            )
            return self._get_default_response()

    def _build_user_prompt(self, num_images: int) -> str:
        base_prompt = self.config.prompt

        if num_images == 1:
            suffix = "You will be provided with an original image of a tire."
        elif num_images == 2:
            suffix = (
                "You will be provided with an original image of a tire and an unwrapped image of the same tire. "
                "Use both images to increase your accuracy."
            )
        else:
            suffix = f"You will be provided with {num_images} images of the tire. Use all images to increase your accuracy."

        return f"{base_prompt} {suffix}"

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

        for example in self.few_shot_examples:
            user_content = [{"type": "text", "text": example["user"]}]

            for image_data in example["images"]:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": image_data}}
                )

            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": example["assistant"]})

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

    def _get_llm_response(self, file_inputs: list[str], user_prompt: str) -> str:
        result = ""
        messages = self._build_messages(file_inputs, user_prompt)

        stream = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            stream=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_completion_tokens,
            presence_penalty=self.config.presence_penalty,
            frequency_penalty=self.config.frequency_penalty,
            extra_body={"provider": {"only": ["nebius"]}},
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                result += chunk.choices[0].delta.content

        self.logger.info(f"LLM response: {result}")
        return result

    async def _async_get_llm_response(
        self, file_inputs: list[str], user_prompt: str
    ) -> str:
        result = ""
        messages = self._build_messages(file_inputs, user_prompt)

        stream = await self.async_client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            stream=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_completion_tokens,
            presence_penalty=self.config.presence_penalty,
            frequency_penalty=self.config.frequency_penalty,
            extra_body={"provider": {"only": ["nebius"]}},
        )

        async for chunk in stream:
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
