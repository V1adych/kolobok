from pathlib import Path
import pickle as pkl
import asyncio

import cv2

from tqdm import tqdm

from tire_vision.text.preprocessor.model import SidewallSegmentator
from tire_vision.text.preprocessor.unwrapper import SidewallUnwrapper
from tire_vision.text.ocr.pipeline import OCRPipeline
from tire_vision.text.index.pipeline import IndexPipeline
from tire_vision.config import (
    SidewallSegmentatorConfig,
    SidewallUnwrapperConfig,
    OCRConfig,
    IndexConfig,
)

import logging

logging.basicConfig(level=logging.INFO)


def get_cached_results(
    pkl_path: str,
    segmentator: SidewallSegmentator,
    unwrapper: SidewallUnwrapper,
    input_dir: Path,
):
    if Path(pkl_path).exists():
        logging.info(f"Loading cached results from {pkl_path}")
        with open(pkl_path, "rb") as f:
            return pkl.load(f)

    logging.info(f"No cached results found, processing from scratch")
    results = []

    for img_path in tqdm(sorted(list(input_dir.iterdir()))):
        logging.info(f"Processing {img_path}")
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = segmentator.detect(img)
        unwrapped = unwrapper.get_unwrapped_tire(img, mask)
        unwrapped = cv2.cvtColor(unwrapped, cv2.COLOR_RGB2BGR)

        images = [img, unwrapped]
        results.append(
            {
                "img_path": str(img_path),
                "images": images,
            }
        )

    with open(pkl_path, "wb") as f:
        pkl.dump(results, f)

    return results

query = """
select 
m.id as model_id,
m.name as model_name,
b.id as brand_id,
b.name as brand_name,
SIMILARITY_SCORE(b.name, 'KAMA') as similarity_score
from models as m
inner join models as b on m.parent_id = b.id
order by similarity_score desc
limit 10
"""

def main():
    cfg_segmentator = SidewallSegmentatorConfig()
    cfg_unwrapper = SidewallUnwrapperConfig()
    cfg_ocr = OCRConfig()
    cfg_index = IndexConfig()
    cfg_index.db_host = "localhost"

    model = SidewallSegmentator(cfg_segmentator)
    unwrapper = SidewallUnwrapper(cfg_unwrapper)
    ocr = OCRPipeline(cfg_ocr)
    index = IndexPipeline(cfg_index)
    db = index.database


    input_dir = Path("/Users/n-zagainov/kolobok/ml/data/annotations")

    results = get_cached_results(
        pkl_path="results.pkl",
        segmentator=model,
        unwrapper=unwrapper,
        input_dir=input_dir,
    )

    # print(db.execute_query(query))

    # Create a dictionary to track which coroutine corresponds to which image path
    coroutines = []
    img_paths = []
    
    for result in results:
        images = result["images"]
        img_path = result["img_path"]
        # Store the coroutine
        coroutines.append(ocr.async_extract_tire_info(images))
        # Store the corresponding image path
        img_paths.append(img_path)

    # Process coroutines in batches of 5
    async def run_coroutines_in_batches():
        all_results = []
        batch_size = 5
        
        for i in range(0, len(coroutines), batch_size):
            batch_coroutines = coroutines[i:i+batch_size]
            batch_results = await asyncio.gather(*batch_coroutines)
            all_results.extend(batch_results)
            print(f"Completed batch {i//batch_size + 1}/{(len(coroutines) + batch_size - 1)//batch_size}")
        
        return all_results
    
    # Run the async function that processes coroutines in batches
    ocr_results = asyncio.run(run_coroutines_in_batches())

    # Print results with their corresponding image paths
    results = []
    for img_path, ocr_result in zip(img_paths, ocr_results):
        print(f"Image: {img_path}")
        print(f"OCR Result: {ocr_result}")
        print("-" * 50)
        results.append({
            "img_path": img_path,
            "ocr_result": ocr_result
        })

    with open("ocr_results.pkl", "wb") as f:
        pkl.dump(results, f)


if __name__ == "__main__":
    main()
