import os
from typing import List, Optional
from PIL import Image, UnidentifiedImageError
import base64
import io

import numpy as np
import cv2
from fastapi import Form, Security, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from tire_vision.thread.pipeline import TireThreadPipeline, TireThreadPipelineResult
from tire_vision.thread.studs.pipeline import Stud
from tire_vision.text.pipeline import TireAnnotationPipeline, AnnotationResult
from tire_vision.config import TireVisionConfig, CLASS_COLORS
from tire_vision.options import TireThreadPipelineOptions, TireAnnotationPipelineOptions

from logs_manager import log_wrapper


cfg = TireVisionConfig()
thread_pipeline = TireThreadPipeline(cfg.thread_pipeline_config)
annotation_pipeline = TireAnnotationPipeline(cfg.annotation_pipeline_config)


bearer_scheme = HTTPBearer(
    scheme_name="Bearer",
    description="Bearer token authentication",
)

API_TOKEN = os.environ["API_TOKEN"]


def parse_thread_options(
    options: Optional[str] = Form(
        None,
        description=(
            "JSON-encoded TireThreadPipelineOptions (confidence_threshold, nms_iou_threshold, max_detections, padding_frac)."
        ),
    ),
) -> Optional[TireThreadPipelineOptions]:
    if options is None:
        return None
    return TireThreadPipelineOptions.model_validate_json(options)


def parse_annotation_options(
    options: Optional[str] = Form(
        None,
        description=("JSON-encoded TireAnnotationPipelineOptions (ocr and index)."),
    ),
) -> Optional[TireAnnotationPipelineOptions]:
    if options is None:
        return None
    return TireAnnotationPipelineOptions.model_validate_json(options)


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
):
    """
    Ensure Authorization: Bearer <token> is present and valid.
    """
    token = credentials.credentials
    if credentials.scheme.lower() != "bearer" or token != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


@log_wrapper
def get_thread_stats(image: np.ndarray, options: Optional[TireThreadPipelineOptions] = None) -> TireThreadPipelineResult:
    result = thread_pipeline(image, options=options)

    return result


@log_wrapper
def extract_tire_info(image: np.ndarray, options: Optional[TireAnnotationPipelineOptions] = None) -> AnnotationResult:
    result = annotation_pipeline(image, options=options)

    return result


def add_annotations(image: np.ndarray, annotations: List[Stud]) -> np.ndarray:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for annotation in annotations:
        x, y, w, h = annotation.box
        color = CLASS_COLORS[annotation.label_id]
        cv2.rectangle(image_bgr, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb


def validate_image(b64_data: str) -> None:
    try:
        raw = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(raw))
        img.verify()
    except (base64.binascii.Error, UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Image is corrupted or not valid")


def validate_image_bytes(image_bytes: bytes) -> None:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Image is corrupted or not valid")


def numpy_to_base64(image: np.ndarray) -> str:
    buffered = io.BytesIO()
    Image.fromarray(image).save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
