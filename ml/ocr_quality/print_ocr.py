from pathlib import Path
import asyncio
import json
import base64
import io
import httpx
import time

from rapidfuzz import fuzz
import cv2
import polars as pl
from tqdm import tqdm

from tire_vision.text.pipeline import TireAnnotationPipeline
from tire_vision.config import TireVisionConfig

import logging


logging.basicConfig(level=logging.CRITICAL, force=True)


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


async def get_prediction(
    client: httpx.AsyncClient,
    image: cv2.typing.MatLike,
    endpoint: str,
    token: str,
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        start_time = time.time()
        image_bytes = get_image_bytes(image)
        headers = {"Authorization": f"Bearer {token}"}
        data = {"image": image_bytes, "model": "ocr"}
        try:
            response = await client.post(endpoint, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            end_time = time.time()
            return response.json(), end_time - start_time
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error occurred: {e}")
            return None, 0
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None, 0


async def main():
    input_dir = Path("/Users/n-zagainov/kolobok/ml/data/annotations")
    gt_paths = Path("/Users/n-zagainov/kolobok/ml/data/annotations_processed")

    # base_url = "http://localhost:8000"
    # base_url = "https://tire-vision.duckdns.org"
    # base_url = "http://51.250.41.44:8000"
    base_url = "http://193.168.196.143:8000"
    # token = "kolobok_token"
    token = "a2400743-8a61-4bcc-82d7-ca3fc160d9f4"
    endpoint = f"{base_url}/api/v1/extract_information"

    cfg = TireVisionConfig()

    pipeline = TireAnnotationPipeline(
        config=cfg.annotation_pipeline_config
    )

    input_names = list(map(lambda x: x.stem, input_dir.iterdir()))
    input_names.sort()

    images_to_process = []
    gts_to_process = []

    for name in tqdm(input_names, desc="Preparing data"):
        img_path = input_dir / f"{name}.jpg"
        gt_path = gt_paths / f"{name}.json"

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        gt_data_raw = read_json(gt_path)

        gt_data_raw = {
            "model": int(gt_data_raw["model"]) if gt_data_raw["model"] is not None else None,
            "brand": int(gt_data_raw["brand"]) if gt_data_raw["brand"] is not None else None,
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
    semaphore = asyncio.Semaphore(5)

    async def get_and_process(
        client: httpx.AsyncClient,
        image: cv2.typing.MatLike,
        gt_data: dict,
        endpoint: str,
        token: str,
        semaphore: asyncio.Semaphore,
    ):
        output, exec_time = await get_prediction(client, image, endpoint, token, semaphore)

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


    async with httpx.AsyncClient() as client:
        tasks = [
            get_and_process(client, image, gt, endpoint, token, semaphore)
            for image, gt in zip(images_to_process[:1], gts_to_process)
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
        print(df)
        print("\nAll metrics:")
        print(df)
        print("\nMean metrics:")
        print(df.select(pl.exclude("name", "raw_output")).mean())

        with open("ocr_results.json", "w") as f:
            json.dump(all_results, f)


if __name__ == "__main__":
    asyncio.run(main())
