from __future__ import annotations

from typing import Optional, Tuple, List
from pydantic import BaseModel, Field
from datetime import datetime

from tire_vision.options import TireThreadPipelineOptions, TireAnnotationPipelineOptions


class ThreadImageRequest(BaseModel):
    image: str = Field(description="base64 encoded image")
    thread_options: Optional[TireThreadPipelineOptions] = Field(
        None,
        description=(
            "Thread options. confidence_threshold: smaller → more detections; padding_frac: crop padding around tire."
        ),
    )


class AnnotationImageRequest(BaseModel):
    image: str = Field(description="base64 encoded image")
    annotation_options: Optional[TireAnnotationPipelineOptions] = Field(
        None,
        description=("OCR and index options (model_name, temperature, top_p; result limits)."),
    )


class PerfStats(BaseModel):
    request_received_timestamp: str = Field(description="Timestamp of request reception")
    request_completed_timestamp: str = Field(description="Timestamp of request completion")
    total_time_seconds: float = Field(description="Total time taken to process the request")

    @staticmethod
    def default() -> PerfStats:
        return PerfStats(
            request_received_timestamp=datetime.now().isoformat(),
            request_completed_timestamp=datetime.now().isoformat(),
            total_time_seconds=0,
        )


class Stud(BaseModel):
    box: Tuple[int, int, int, int] = Field(description="Bounding box of a stud (cx, cy, w, h)")
    label_id: int = Field(description="Label ID of a stud (0: broken, 1: healthy)")
    label: str = Field(description="Label of a stud (broken or healthy)")


class TireThreadPipelineResult(BaseModel):
    depth: float
    studs: List[Stud]


class ThreadAnalysisResponse(BaseModel):
    thread_depth: float = Field(description="Estimated depth of a thread")
    studs: List[Stud] = Field(description="List of detected studs")
    image: str = Field(description="base64 encoded image with annotations")
    perf_stats: PerfStats = Field(PerfStats.default(), description="Performance statistics")


class IndexResult(BaseModel):
    model_id: int = Field(description="id of a tire model in the database")
    model_name: str = Field(description="Name of the tire model (exact match from the database)")
    candidate_model_name: str = Field(description="Name of the tire model (raw OCR result)")
    candidate_model_score: float = Field(
        description="Similarity score between the raw OCR result and the exact match from the database"
    )
    brand_id: int = Field(description="id of a tire brand in the database")
    brand_name: str = Field(description="Name of the tire brand (exact match from the database)")
    candidate_brand_name: str = Field(description="Name of the tire brand (raw OCR result)")
    candidate_brand_score: float = Field(
        description="Similarity score between the raw OCR result and the exact match from the database"
    )
    combined_score: float = Field(description="Combined similarity score of the model and the brand")


class AnnotationResult(BaseModel):
    strings: List[str]
    tire_size: str
    index_results: List[IndexResult]


class ExtractInformationResponse(BaseModel):
    strings: List[str] = Field(description="Raw OCR results")
    tire_size: str = Field(description="Detected tire size")
    index_results: List[IndexResult] = Field(description="List of index results")
    perf_stats: PerfStats = Field(PerfStats.default(), description="Performance statistics")
