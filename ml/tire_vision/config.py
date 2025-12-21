from dataclasses import dataclass, field
import os
from typing import Literal, Tuple, List
import multiprocessing

import onnxruntime as ort

from tire_vision.options import (
    ThreadSegmentatorOptions,
    StudPipelineOptions,
    SidewallSegmentatorOptions,
    SidewallUnwrapperOptions,
    OCROptions,
    IndexOptions,
)


DEVICE = "cpu"

META_TO_LABEL_MAPPING = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
}

LABEL_MAPPING = {
    0: "broken",
    1: "healthy",
}

META_MAPPING = {
    0: "absent",
    1: "broken",
    2: "floating",
    3: "healthy",
}

META_COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 255),
    2: (255, 0, 0),
    3: (0, 255, 0),
}

SYSTEM_OCR_PROMPT = """You are an expert OCR model specializing in reading text from images of tires. 
Your tasks will involve reading, parsing, extracting, and analyzing information from the images of wheel tires

Your primary focus should be on the following properties of the tire:
- Brand
- Model
- Size in format <width>/<aspect_ratio>R<diameter> <load_index><speed_index> or just <width>/<aspect_ratio>R<diameter>

Do not include other redundant information, such as:
- external links (example: 'www.vatti-tyres.com')
- country of origin (example: 'Made in China')
- other information that is not really relevant to the tire identification. usually low-font text (examples: 'DOT 1H2 RYCHU 2323')

When constructing the response, try to include all the information you found (and at the same time, relevant).
Example:
If you find a string consisting of multiple words, return all possible combinations of them.

string on tire: 'GitiComfort F22'
response must include: ['GitiComfort F22', 'GitiComfort', 'F22']

string on tire: '215/50R17 95V' (size)
response must include: ['215/50R17 95V', '215/50R17', '95V']

string on tire: 'KAMA EURO-236' (brand and model, but you don't know which is which)
response must include: ['KAMA EURO-236', 'KAMA', 'EURO', 'EURO-236', 'KAMA EURO', '236']

The examples above are just examples, do not follow them strictly.
Do not make up any information, return only what you can see."""

OCR_PROMPT = """Your task is to extract text from the provided image(s) of a tire.
Extract all visible text from the image(s) that might be related to tire information (model, brand, size)
Particularly important to emphasize the tire size string. You should return it separately.
Present the extracted text as a JSON object with keys "strings", "tire_size".
- "strings" is a list of all (relevant) text strings found on the tire.
- "tire_size" is the tire size string found on the tire in format <width>/<aspect_ratio>R<diameter> <load_index><speed_index> or just <width>/<aspect_ratio>R<diameter> (if load_index and speed_index are not present)
Do not include any reasoning or explanations, only the final JSON object.

Example of a valid response:
```json
{
    "strings": ["MICHELIN", "Pilot Sport 4 S", "Pilot Sport", "Pilot", "Sport", "Sport 4 S", "245/35ZR20", "95Y", "245/35ZR20 95Y"],
    "tire_size": "245/35ZR20 95Y"
}
```
"""

ORIGINAL_PROMPT = "You will receive a single image of a tire"
UNWRAP_PROMPT = "You will receive an unwrapped image of a tire (unwrap performed for better readability of the text)"
BOTH_PROMPT = (
    "You will receive both an original image of a tire and its unwrapped version (unwrap performed for better readability of the text). "
    "Use both images to extract the required information."
)

num_gunicorn_workers = int(os.environ.get("GUNICORN_WORKERS", "1"))
ort_opts = ort.SessionOptions()
ort_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_opts.enable_cpu_mem_arena = True
ort_opts.enable_mem_pattern = True
cores = multiprocessing.cpu_count()
ort_opts.intra_op_num_threads = max(1, cores // num_gunicorn_workers)
ort_opts.inter_op_num_threads = 1

ort_providers = ["CPUExecutionProvider"]


@dataclass(frozen=True)
class ThreadSegmentatorConfig:
    thread_segmentator_onnx: str = "onnx/thread_segmentator.onnx"
    resize_shape: Tuple[int, int] = (512, 512)
    options: ThreadSegmentatorOptions = field(default_factory=ThreadSegmentatorOptions)


@dataclass(frozen=True)
class StudPipelineConfig:
    spike_detector_onnx: str = "onnx/stud_detector.onnx"
    resize_shape: Tuple[int, int] = (560, 560)
    options: StudPipelineOptions = field(default_factory=StudPipelineOptions)


@dataclass(frozen=True)
class DepthRegressorConfig:
    depth_regressor_onnx: str = "onnx/depth_regressor.onnx"
    resize_shape: Tuple[int, int] = (512, 512)


@dataclass(frozen=True)
class SidewallSegmentatorConfig:
    sidewall_segmentator_onnx: str = "onnx/sidewall_segmentator.onnx"
    resize_shape: Tuple[int, int] = (512, 512)
    options: SidewallSegmentatorOptions = field(default_factory=SidewallSegmentatorOptions)


@dataclass(frozen=True)
class SidewallUnwrapperConfig:
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    rectify_aspect_ratio_threshold: float = 1.1
    mask_postprocess_ksize: int = 21
    concat_strip: bool = True
    options: SidewallUnwrapperOptions = field(default_factory=SidewallUnwrapperOptions)


@dataclass(frozen=True)
class OCRConfig:
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = os.environ["OPENROUTER_API_KEY"]
    system_prompt: str = SYSTEM_OCR_PROMPT
    prompt: str = OCR_PROMPT
    options: OCROptions = field(default_factory=OCROptions)


@dataclass(frozen=True)
class IndexConfig:
    db_host: str = os.environ.get("DB_HOST", "mysql_db")
    db_port: int = int(os.environ.get("DB_PORT", "3306"))
    db_name: str = os.environ["MYSQL_DATABASE"]
    db_user: str = "root"
    db_password: str = os.environ["MYSQL_ROOT_PASSWORD"]
    table_name: str = "models"
    table_cache_path: str = "models.parquet"
    table_cache_ttl_seconds: int = float("inf")
    options: IndexOptions = field(default_factory=IndexOptions)


@dataclass(frozen=True)
class TireAnnotationPipelineConfig:
    sidewall_segmentator_config: SidewallSegmentatorConfig = SidewallSegmentatorConfig()
    sidewall_unwrapper_config: SidewallUnwrapperConfig = SidewallUnwrapperConfig()
    ocr_config: OCRConfig = OCRConfig()
    index_config: IndexConfig = IndexConfig()

    max_image_size: int = 2048
    tire_size_regex: List[str] = field(
        default_factory=lambda: [
            r"\d{3}/\d{2}[Rr]\d{2} \d{2}[A-Za-z]{2}",
            r"\d{3}/\d{2}[Rr]\d{2}",
        ]
    )
    image_composition_strategy: Literal["unwrapped", "both"] = "both"

    original_prompt: str = ORIGINAL_PROMPT
    unwrap_prompt: str = UNWRAP_PROMPT
    both_prompt: str = BOTH_PROMPT


@dataclass(frozen=True)
class TireThreadPipelineConfig:
    thread_segmentator_config: ThreadSegmentatorConfig = ThreadSegmentatorConfig()
    stud_pipeline_config: StudPipelineConfig = StudPipelineConfig()
    depth_regressor_config: DepthRegressorConfig = DepthRegressorConfig()


@dataclass(frozen=True)
class TireVisionConfig:
    thread_pipeline_config: TireThreadPipelineConfig = TireThreadPipelineConfig()
    annotation_pipeline_config: TireAnnotationPipelineConfig = TireAnnotationPipelineConfig()
