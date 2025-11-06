from pathlib import Path
import asyncio
import json
import base64
import io
import httpx
import time
import os

from rapidfuzz import fuzz
import cv2
import polars as pl
from tqdm import tqdm

from tire_vision.text.pipeline import TireAnnotationPipeline
from tire_vision.config import TireVisionConfig

import logging
from dataclasses import dataclass
from typing import Optional

import tyro


logging.basicConfig(level=logging.INFO, force=True)


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def check_correct(gt: dict[str, object], candidates: list[dict[str, object]]):
    result = dict(
        absolutely_correct=0,
        max_ratio_brand=0,
        max_ratio_model=0,
    )
    if not candidates:
        return result

    for candidate in candidates:
        if (
            candidate["model_id"] == gt["model_id"]
            and candidate["brand_id"] == gt["brand_id"]
        ):
            result["absolutely_correct"] = 1
            result["max_ratio_brand"] = 1
            result["max_ratio_model"] = 1
            return result

        if candidate["brand_id"] == gt["brand_id"]:
            result["absolutely_correct"] = 0.5
            result["max_ratio_brand"] = 1

        if candidate["model_id"] == gt["model_id"]:
            result["absolutely_correct"] = 0.5
            result["max_ratio_model"] = 1

        ratio_brand = fuzz.ratio(candidate["brand_name"], gt["brand_name"]) / 100
        ratio_model = fuzz.ratio(candidate["model_name"], gt["model_name"]) / 100
        result["max_ratio_brand"] = max(result["max_ratio_brand"], ratio_brand)
        result["max_ratio_model"] = max(result["max_ratio_model"], ratio_model)

    return result


def get_image_bytes(image: cv2.typing.MatLike) -> str:
    _, buffer = cv2.imencode(".jpg", image)
    img_bytes = io.BytesIO(buffer)
    return base64.b64encode(img_bytes.read()).decode("utf-8")


def get_image_bytes_v2(image: cv2.typing.MatLike) -> bytes:
    """Encodes a cv2 image to jpg bytes."""
    _, buffer = cv2.imencode(".jpg", image)
    return buffer.tobytes()


async def get_prediction(
    client: httpx.AsyncClient,
    image: cv2.typing.MatLike,
    endpoint: str,
    token: str,
    semaphore: asyncio.Semaphore,
    timeout: float,
):
    async with semaphore:
        start_time = time.time()
        image_bytes = get_image_bytes(image)
        headers = {"Authorization": f"Bearer {token}"}
        data = {"image": image_bytes, "annotation_options": {"ocr_options": {"model_name": "qwen/qwen3-vl-30b-a3b-instruct"}}}
        try:
            response = await client.post(
                endpoint, json=data, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            end_time = time.time()
            return response.json(), end_time - start_time
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error occurred: {e}")
            return None, 0
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None, 0


async def get_prediction_v2(
    client: httpx.AsyncClient,
    image: cv2.typing.MatLike,
    endpoint: str,
    token: str,
    semaphore: asyncio.Semaphore,
    timeout: float,
):
    """Sends an image to a v2 endpoint using multipart/form-data."""
    async with semaphore:
        start_time = time.time()
        image_bytes = get_image_bytes_v2(image)
        headers = {"Authorization": f"Bearer {token}"}
        files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
        data = {"annotation_options": {"ocr_options": {"model_name": "qwen/qwen3-vl-30b-a3b-instruct"}}}
        try:
            response = await client.post(
                endpoint, files=files, headers=headers, timeout=timeout, data=data
            )
            response.raise_for_status()
            end_time = time.time()
            return response.json(), end_time - start_time
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error occurred: {e}")
            return None, 0
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None, 0


@dataclass
class Args:
    input_dir: str
    gt_dir: str
    url: str = "http://localhost:8000/api/v1/extract_information"
    token: Optional[str] = None
    concurrency: int = 4
    limit: Optional[int] = None
    verify_ssl: bool = True
    multipart: bool = False
    timeout: float = 30.0
    output_json: Optional[str] = "ocr_results.json"


async def main():
    args = tyro.cli(Args)
    input_dir = Path(args.input_dir)
    gt_paths = Path(args.gt_dir)
    endpoint = args.url

    token = args.token or os.environ["API_TOKEN"]

    cfg = TireVisionConfig()

    pipeline = TireAnnotationPipeline(config=cfg.annotation_pipeline_config)

    input_names = list(map(lambda x: x.stem, input_dir.iterdir()))
    input_names.sort()
    if args.limit is not None:
        input_names = input_names[: args.limit]

    images_to_process = []
    gts_to_process = []

    for name in tqdm(input_names, desc="Preparing data"):
        img_path = input_dir / f"{name}.jpg"
        gt_path = gt_paths / f"{name}.json"

        # image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        image = cv2.imread(str(img_path))
        gt_data_raw = read_json(gt_path)

        gt_data_raw = {
            "model": int(gt_data_raw["model"])
            if gt_data_raw["model"] is not None
            else None,
            "brand": int(gt_data_raw["brand"])
            if gt_data_raw["brand"] is not None
            else None,
        }

        if gt_data_raw["model"] is None or gt_data_raw["brand"] is None:
            continue

        model_name = (
            pipeline.index.database.table.filter(pl.col("id") == gt_data_raw["model"])
            .select(pl.col("name").alias("model_name"))
            .to_dicts()[0]["model_name"]
        )
        brand_name = (
            pipeline.index.database.table.filter(pl.col("id") == gt_data_raw["brand"])
            .select(pl.col("name").alias("brand_name"))
            .to_dicts()[0]["brand_name"]
        )

        gt_data = {
            "name": name,
            "model_id": gt_data_raw["model"],
            "brand_id": gt_data_raw["brand"],
            "model_name": model_name,
            "brand_name": brand_name,
        }
        images_to_process.append(image)
        gts_to_process.append(gt_data)

    all_results = []
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def get_and_process(
        client: httpx.AsyncClient,
        image: cv2.typing.MatLike,
        gt_data: dict,
        endpoint: str,
        token: str,
        semaphore: asyncio.Semaphore,
    ):
        # if args.multipart:
        #     output, exec_time = await get_prediction_v2(
        #         client, image, endpoint, token, semaphore, args.timeout
        #     )
        # else:
        output, exec_time = await get_prediction(
            client, image, endpoint, token, semaphore, args.timeout
        )

        if output is None:
            return None

        result = check_correct(gt_data, output.get("index_results", []))
        result["gt_model"] = gt_data["model_name"]
        result["gt_brand"] = gt_data["brand_name"]
        result["is_tire_size_nonempty"] = int(bool(output.get("tire_size")))
        result["raw_output"] = output
        result["execution_time"] = exec_time
        result_with_name = {"name": gt_data["name"], **result}

        return result_with_name

    async with httpx.AsyncClient(verify=args.verify_ssl, http2=False) as client:
        tasks = [
            get_and_process(client, image, gt, endpoint, token, semaphore)
            for image, gt in zip(images_to_process, gts_to_process)
        ]

        for f in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Getting predictions",
        ):
            result = await f
            if result:
                all_results.append(result)

    if all_results:
        df = pl.DataFrame(all_results)
        print("\nAll metrics:")
        print(df)
        print("\nMean metrics:")
        print(df.select(pl.exclude("name", "raw_output")).mean())

        output_path = (
            Path(args.output_json) if args.output_json else Path("ocr_results.json")
        )
        with open(output_path, "w") as f:
            json.dump(all_results, f)


if __name__ == "__main__":
    asyncio.run(main())
