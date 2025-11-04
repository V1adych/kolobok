from typing import Optional, Tuple, List
from pydantic import BaseModel, Field, ConfigDict

from tire_vision.options import TireThreadPipelineOptions, TireAnnotationPipelineOptions


class ThreadImageRequest(BaseModel):
    image: str
    thread_options: Optional[TireThreadPipelineOptions] = Field(
        None,
        description=(
            "Thread options. confidence_threshold: smaller → more detections; padding_frac: crop padding around tire."
        ),
    )


class AnnotationImageRequest(BaseModel):
    image: str
    annotation_options: Optional[TireAnnotationPipelineOptions] = Field(
        None,
        description=(
            "OCR and index options (model_name, temperature, top_p; result limits)."
        ),
    )


class PerfStats(BaseModel):
    request_received_timestamp: str
    request_completed_timestamp: str
    total_time_seconds: float


class StudDetection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    box: Tuple[int, int, int, int]
    class_: int = Field(alias="class")
    label: str


class ThreadAnalysisResponse(BaseModel):
    success: int
    thread_depth: Optional[float] = None
    studs: Optional[List[StudDetection]] = None
    image: Optional[str] = None
    detail: Optional[str] = None
    perf_stats: Optional[PerfStats] = None


class IndexResult(BaseModel):
    model_id: int
    model_name: str
    candidate_model_name: str
    candidate_model_score: float
    brand_id: int
    brand_name: str
    candidate_brand_name: str
    candidate_brand_score: float
    combined_score: float


class ExtractInformationResponse(BaseModel):
    strings: List[str]
    tire_size: str
    index_results: List[IndexResult]
    perf_stats: Optional[PerfStats] = None
