from typing import Tuple, List, Literal
from dataclasses import field
from pydantic import BaseModel


class ThreadSegmentatorOptions(BaseModel):
    confidence_threshold: float = 0.5
    padding_frac: float = 0.01
    min_tire_pixels: int = 96


class StudPipelineOptions(BaseModel):
    max_detections: int = 300
    nms_iou_threshold: float = 0.15
    confidence_threshold: float = 0.25


class SidewallSegmentatorOptions(BaseModel):
    confidence_threshold: float = 0.5


class SidewallUnwrapperOptions(BaseModel):
    polar_unwrap_size: Tuple[int, int] = (1000, 2500)


class OCROptions(BaseModel):
    model_name: str = "qwen/qwen3-vl-32b-instruct"
    providers_list: List[str] = field(default_factory=lambda: ["parasail"])
    top_p: float = 0.95
    temperature: float = 0.7
    presence_penalty: float = 0
    frequency_penalty: float = 0
    max_completion_tokens: int = 1024


class IndexOptions(BaseModel):
    max_query_results: int = 10
    max_brand_matches: int = 20
    max_model_matches: int = 50
    max_distinct_matches: int = 3
    brand_model_match_bonus: float = 0.1
    similarity_metric: Literal["levenshtein", "jaro_winkler"] = "jaro_winkler"
    comb_metric: Literal[
        "product",
        "arithmetic_mean",
        "harmonic_mean",
        "geometric_mean",
        "euclidean",
    ] = "arithmetic_mean"


class TireAnnotationPipelineOptions(BaseModel):
    sidewall_segmentator_options: SidewallSegmentatorOptions = (
        SidewallSegmentatorOptions()
    )
    sidewall_unwrapper_options: SidewallUnwrapperOptions = SidewallUnwrapperOptions()
    ocr_options: OCROptions = OCROptions()
    index_options: IndexOptions = IndexOptions()


class TireThreadPipelineOptions(BaseModel):
    thread_segmentator_options: ThreadSegmentatorOptions = ThreadSegmentatorOptions()
    stud_pipeline_options: StudPipelineOptions = StudPipelineOptions()
