from dataclasses import dataclass, field
import os
from typing import Literal, Tuple, List
import multiprocessing

import onnxruntime as ort


DEVICE = "cpu"


CLASS_MAPPING = {
    0: "good",
    1: "bad",
}

CLASS_COLORS = {
    "good": (0, 255, 0),
    "bad": (255, 0, 0),
}

SYSTEM_OCR_PROMPT = """You are "Tire-VLM-OCR", an expert model for extracting legible, in-frame text from photographs of wheel-and-tire assemblies.

PRIMARY GOAL
Return ALL the text that can be clearly seen on the tire
Key attributes, (Must be always present in your response):

1. Tire size – forms like "<width>/<aspect_ratio>R<diameter>( <load_index><speed_index>)?"
   - Example: "245/35ZR20 95Y".
2. Brand name – e.g. "MICHELIN", "KAMA", "NOKIAN TYRES".
3. Model or sub-brand – e.g. "Pilot Sport 4 S", "EURO-236".

HALLUCINATION RULE
If a glyph or word is ambiguous or clipped, leave it out.  Never invent characters.

TOKEN VARIANTS (de-dup safe)
When a string contains multiple words, include every meaningful combination once (drop case-insensitive duplicates).

Examples
--------
Text on tire: "GitiComfort F22"  
Return: ["GitiComfort F22", "GitiComfort", "F22"]

Text on tire: "KAMA EURO-236"  
Return: ["KAMA EURO-236", "KAMA", "EURO-236", "EURO"]

Text on tire: "Continental ContiWinterContact TS 850 P"  
Return: ["Continental ContiWinterContact TS 850 P",
         "Continental",
         "ContiWinterContact TS 850 P",
         "ContiWinterContact",
         "TS 850 P",
         "TS",
         "850",
         "P"]

Return NOTHING except what is visible (do just not repeat examples)."""

OCR_PROMPT = """Extract every string on the tire.

Output one JSON object with the following keys:

- "strings": string[]   // unique, relevance-filtered tokens and token-groups (see rules)
- "tire_size": string   // the exact size string (regex: \d{3}/\d{2}R\d{2}\w?(?:\s+\d{2,3}\w)?)

Hint: "tire_size" string should be among "strings"

Constraints
-----------
1. Return the JSON object—no prose, no comments.
2. If multiple size strings occur, choose the MOST complete (includes load & speed indexes).
3. The order of "strings" is arbitrary; duplicates (case-insensitive) are forbidden.
4. Preserve original spacing, hyphens, and letter-case exactly as read.

Examples
========

Example A — common size with load/speed
--------------------------------------
*Visible text:*  
"MICHELIN", "Pilot Sport 4 S", "245/35ZR20 95Y"

```json
{
  "strings": [
    "MICHELIN",
    "Pilot Sport 4 S",
    "Pilot Sport",
    "Pilot",
    "Sport",
    "Sport 4 S",
    "245/35ZR20",
    "95Y",
    "245/35ZR20 95Y"
  ],
  "tire_size": "245/35ZR20 95Y"
}
```
"""

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
    confidence_threshold: float = 0.5
    resize_shape: Tuple[int, int] = (512, 512)
    padding_frac: float = 0.01
    min_tire_pixels: int = 96


@dataclass(frozen=True)
class SpikePipelineConfig:
    spike_segmentator_onnx: str = "onnx/spike_segmentator.onnx"
    spike_classifier_onnx: str = "onnx/spike_classifier.onnx"
    confidence_threshold: float = 0.5
    resize_shape: Tuple[int, int] = (512, 512)
    erosion_iterations: int = 0
    dilation_iterations: int = 0
    crop_size: int = 32


@dataclass(frozen=True)
class DepthRegressorConfig:
    depth_regressor_onnx: str = "onnx/depth_regressor.onnx"
    resize_shape: Tuple[int, int] = (512, 512)


@dataclass(frozen=True)
class SidewallSegmentatorConfig:
    sidewall_segmentator_onnx: str = "onnx/sidewall_segmentator.onnx"
    confidence_threshold: float = 0.5
    resize_shape: Tuple[int, int] = (512, 512)


@dataclass(frozen=True)
class SidewallUnwrapperConfig:
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    concat_strip: bool = True
    mask_postprocess_ksize: int = 21
    rectify_aspect_ratio_threshold: float = 1.1
    polar_dsize: Tuple[int, int] = (1000, 2500)
    concat_border_size: int = 5


@dataclass(frozen=True)
class OCRConfig:
    model_name: str = "qwen/qwen2.5-vl-72b-instruct"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = os.environ["OPENROUTER_API_KEY"]
    system_prompt: str = SYSTEM_OCR_PROMPT
    prompt: str = OCR_PROMPT
    providers_list: List[str] = field(default_factory=lambda: ["parasail"])
    top_p: float = 0.95
    temperature: float = 0.7
    presence_penalty: float = 0
    frequency_penalty: float = 0
    max_completion_tokens: int = 1024
    max_image_size: int = 1536


@dataclass(frozen=True)
class IndexConfig:
    db_host: str = os.environ.get("DB_HOST", "mysql_db")
    db_port: int = int(os.environ.get("DB_PORT", "3306"))
    db_name: str = os.environ["MYSQL_DATABASE"]
    db_user: str = "root"
    db_password: str = os.environ["MYSQL_ROOT_PASSWORD"]
    table_name: str = "models"
    max_query_results: int = 5
    table_cache_path: str = "models.parquet"
    table_cache_ttl_seconds: int = 3600
    similarity_metric: Literal[
        "product",
        "arithmetic_mean",
        "harmonic_mean",
        "geometric_mean",
        "euclidean",
    ] = "arithmetic_mean"


@dataclass(frozen=True)
class TireAnnotationPipelineConfig:
    sidewall_segmentator_config: SidewallSegmentatorConfig = SidewallSegmentatorConfig()
    sidewall_unwrapper_config: SidewallUnwrapperConfig = SidewallUnwrapperConfig()
    ocr_config: OCRConfig = OCRConfig()
    index_config: IndexConfig = IndexConfig()
    max_image_size: int = 448 * 5
    image_composition_strategy: Literal["unwrapped", "both"] = "unwrapped"


@dataclass(frozen=True)
class TireThreadPipelineConfig:
    thread_segmentator_config: ThreadSegmentatorConfig = ThreadSegmentatorConfig()
    spike_pipeline_config: SpikePipelineConfig = SpikePipelineConfig()
    depth_regressor_config: DepthRegressorConfig = DepthRegressorConfig()


@dataclass(frozen=True)
class TireVisionConfig:
    thread_pipeline_config: TireThreadPipelineConfig = TireThreadPipelineConfig()
    annotation_pipeline_config: TireAnnotationPipelineConfig = (
        TireAnnotationPipelineConfig()
    )
