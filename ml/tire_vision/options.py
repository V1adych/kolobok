from typing import Tuple, List, Literal
from pydantic import BaseModel, Field

SimilarityMetric = Literal["levenshtein", "jaro_winkler"]
CombMetric = Literal["product", "arithmetic_mean", "harmonic_mean", "geometric_mean", "euclidean"]


class ThreadSegmentatorOptions(BaseModel):
    confidence_threshold: float = Field(0.5, description="Confidence threshold for tire thread detection. Lower values mean higher sensitivity", le=1, ge=0)
    padding_frac: float = Field(0.01, description="Additional padding around the detected thread. Higher the value, more padding is added around the thread", ge=0)
    min_tire_pixels: int = Field(96, description="Detected regions must have at least this many pixels to be considered a tire", gt=0)


class StudPipelineOptions(BaseModel):
    max_detections: int = Field(300, description="Maximum possible number of studs to detect", gt=0, le=300)
    nms_iou_threshold: float = Field(
        0.15, description="Non-maximum suppression (NMS) IoU threshold for merging overlapping stud detections. Lower values mean higher merging rate", ge=0, le=1
    )
    confidence_threshold: float = Field(0.3, description="Confidence threshold for stud detection. Lower values mean higher sensitivity", ge=0, le=1)


class SidewallSegmentatorOptions(BaseModel):
    confidence_threshold: float = Field(0.5, description="Confidence threshold for tire sidewall detection. Lower values mean higher sensitivity", ge=0, le=1)


class SidewallUnwrapperOptions(BaseModel):
    polar_unwrap_size: Tuple[int, int] = Field(
        (1000, 2500),
        description="Detected sidewall is flattened to an image of this resolution and passed to the OCR. Higher values mean more detailed image (higher OCR quality), but also higher quota consumption and processing time",
    )


class OCROptions(BaseModel):
    model_name: str = Field(
        "qwen/qwen3-vl-30b-a3b-instruct", description="Name of a VLM model to use for OCR from openrouter.ai. Insert any valid VLM model from https://openrouter.ai/models"
    )
    providers_list: List[str] = Field([], description="List of providers to use for OCR. Insert any valid provider from https://openrouter.ai/providers")
    top_p: float = Field(0.95, description="Nucleus sampling parameter for VLM. Higher values mean more diverse completions", gt=0, lt=1)
    temperature: float = Field(0.7, description="Temperature parameter for VLM. Higher values mean more diverse completions", gt=0)
    presence_penalty: float = Field(0, description="Presence penalty for VLM. Higher values mean less likely to repeat same text", ge=0)
    frequency_penalty: float = Field(0, description="Frequency penalty for VLM. Higher values mean less likely to repeat same text", ge=0)
    max_completion_tokens: int = Field(1024, description="Maximum number of tokens to generate in the completion. Lower values ensure faster completion and cheaper usage", ge=0)


class IndexOptions(BaseModel):
    max_query_results: int = Field(10, description="Maximum number of results to return from the database. Higher values mean more results", ge=0)
    max_brand_matches: int = Field(20, description="Maximum number of distinct tire brands to consider. Higher values mean better quality and higher processing time", ge=0)
    max_model_matches: int = Field(50, description="Maximum number of distinct tire models to consider. Higher values mean better quality and higher processing time", ge=0)
    max_distinct_matches: int = Field(
        3, description="Maximum number of distinct matches to consider for each detected string. Lower values mean more diversity in the results", ge=0
    )
    brand_model_match_bonus: float = Field(
        0.1,
        description="How much score is added to pairs of model and brand matches in the database. Higher values mean more likely to return existing brand-model pairs from the database",
        ge=0,
    )
    similarity_metric: SimilarityMetric = Field("jaro_winkler", description="Similarity metric to use for comparing OCR results and database entries")
    comb_metric: CombMetric = Field("arithmetic_mean", description="Function to combine similarity scores of model and brand matches")


class TireAnnotationPipelineOptions(BaseModel):
    sidewall_segmentator_options: SidewallSegmentatorOptions = SidewallSegmentatorOptions()
    sidewall_unwrapper_options: SidewallUnwrapperOptions = SidewallUnwrapperOptions()
    ocr_options: OCROptions = OCROptions()
    index_options: IndexOptions = IndexOptions()


class TireThreadPipelineOptions(BaseModel):
    thread_segmentator_options: ThreadSegmentatorOptions = ThreadSegmentatorOptions()
    stud_pipeline_options: StudPipelineOptions = StudPipelineOptions()
