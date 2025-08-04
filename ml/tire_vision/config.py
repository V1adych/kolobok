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

SYSTEM_OCR_PROMPT = """You are "InternVL-OCR", an expert model for extracting legible, in-frame text from photographs of wheel-and-tire assemblies.

PRIMARY GOAL
Return only text that can be clearly seen in the image AND that helps identify the tire.
Key attributes, in strict order of importance:

1. Tire size – forms like "<width>/<aspect_ratio>R<diameter>( <load_index><speed_index>)?"
   - Accept the variants ZR / RF / XL, etc.  Example: "245/35ZR20 95Y".
2. Brand name – e.g. "MICHELIN", "KAMA".
3. Model or sub-brand – e.g. "Pilot Sport 4 S", "EURO-236".

DO NOT RETURN
- URLs, QR codes, DOT/date codes, "Made in ...", E-markings, warnings, production-plant codes, or tiny embossed text unrelated to identification.

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

QUALITY HINTS
- Prefer high-contrast, in-focus regions.
- Ignore mirrored / upside-down duplicates of the same string.
- Treat hyphen "-" as significant; treat whitespace collapse as delimiter.

Return NOTHING except what is visible."""

OCR_PROMPT = """Extract every clearly legible string on the tire that relates to its identification.

Output one JSON object with the following keys:

- "strings": string[]   // unique, relevance-filtered tokens and token-groups (see rules)
- "tire_size": string   // the exact size string (regex: \d{3}/\d{2}R\d{2}\w?(?:\s+\d{2,3}\w)?)

Constraints
-----------
1. Return ONLY the JSON object—no prose, no comments.
2. If multiple size strings occur, choose the MOST complete (includes load & speed indexes).
3. If no valid size is visible, set "tire_size": "".
4. The order of "strings" is arbitrary; duplicates (case-insensitive) are forbidden.
5. Preserve original spacing, hyphens, and letter-case exactly as read.

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


@dataclass
class ThreadSegmentatorConfig:
    thread_segmentator_onnx: str = "onnx/thread_segmentator.onnx"
    confidence_threshold: float = 0.5
    resize_shape: Tuple[int, int] = (512, 512)
    padding_frac: float = 0.01
    min_tire_pixels: int = 96


@dataclass
class SpikePipelineConfig:
    spike_segmentator_onnx: str = "onnx/spike_segmentator.onnx"
    spike_classifier_onnx: str = "onnx/spike_classifier.onnx"
    confidence_threshold: float = 0.5
    resize_shape: Tuple[int, int] = (512, 512)
    erosion_iterations: int = 0
    dilation_iterations: int = 0
    crop_size: int = 32


@dataclass
class DepthRegressorConfig:
    depth_regressor_onnx: str = "onnx/depth_regressor.onnx"
    resize_shape: Tuple[int, int] = (512, 512)


@dataclass
class SidewallSegmentatorConfig:
    sidewall_segmentator_onnx: str = "onnx/sidewall_segmentator.onnx"
    confidence_threshold: float = 0.5
    resize_shape: Tuple[int, int] = (512, 512)


@dataclass
class SidewallUnwrapperConfig:
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    concat_strip: bool = True
    mask_postprocess_ksize: int = 21
    rectify_aspect_ratio_threshold: float = 1.1
    polar_dsize: Tuple[int, int] = (700, 2000)
    concat_border_size: int = 5


@dataclass
class OCRConfig:
    model_name: str = "opengvlab/internvl3-14b"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = os.environ["OPENROUTER_API_KEY"]
    system_prompt: str = SYSTEM_OCR_PROMPT
    prompt: str = OCR_PROMPT
    providers_list: List[str] = field(default_factory=lambda: [])
    top_p: float = 0.9
    temperature: float = 0.7
    presence_penalty: float = 0
    frequency_penalty: float = 0
    max_completion_tokens: int = 1024


@dataclass
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
    ] = "harmonic_mean"


@dataclass
class TireVisionConfig:
    thread_segmentator_config = ThreadSegmentatorConfig()
    spike_pipeline_config = SpikePipelineConfig()
    depth_regressor_config = DepthRegressorConfig()
    sidewall_segmentator_config = SidewallSegmentatorConfig()
    sidewall_unwrapper_config = SidewallUnwrapperConfig()
    ocr_config = OCRConfig()
    index_config = IndexConfig()
