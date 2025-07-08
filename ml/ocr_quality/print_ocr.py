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
        mask = segmentator.forward(img)
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


def get_cached_ocr_results(pkl_path: str):
    if not Path(pkl_path).exists():
        logging.info(f"No cached OCR results found, processing from scratch")
        return []

    logging.info(f"Loading cached OCR results from {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pkl.load(f)


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

    results = get_cached_ocr_results(
        pkl_path="ocr_results.pkl",
    )

    result = index.get_best_matches(
        [
            "AMTEL",
            "PLANET DC",
            "175/70R13 82H",
            "STEEL BELTED RADIAL",
            "TUBELESS",
            "175/70R13",
            "82H",
        ]
    )

    print(result)

    # for result in results:
    #     img_path = result["img_path"]
    #     strings = result["ocr_result"]["strings"]
    #     print(strings)
    #     break


if __name__ == "__main__":
    main()
