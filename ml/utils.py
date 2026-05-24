import os
from typing import List, Optional
from PIL import Image, UnidentifiedImageError
import base64
import io

import numpy as np
import cv2
from fastapi import Form, Security, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from tire_vision.thread.pipeline import AnalyzedTire, TireThreadPipeline, TireThreadPipelineResult
from tire_vision.text.pipeline import TireAnnotationPipeline, AnnotationResult
from tire_vision.config import TireVisionConfig, STUD_COLORS
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
            "JSON-encoded TireThreadPipelineOptions for tire segmentation, stud detection, and ambiguous stud resolution."
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


def add_annotations(image: np.ndarray, annotations: List[AnalyzedTire]) -> np.ndarray:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w = image_bgr.shape[:2]
    tire_colors = [(255, 0, 0), (0, 180, 255), (255, 0, 255), (0, 255, 0)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_padding = 4
    for idx, annotation in enumerate(annotations):
        x, y, w, h = annotation.box
        tire_color = tire_colors[idx % len(tire_colors)]
        mask = annotation.mask.astype(bool)
        image_bgr[mask] = (0.8 * image_bgr[mask] + 0.2 * np.array(tire_color, dtype=np.uint8)).astype(np.uint8)
        x1 = max(0, x - w // 2)
        y1 = max(0, y - h // 2)
        x2 = min(img_w - 1, x + w // 2)
        y2 = min(img_h - 1, y + h // 2)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), tire_color, 3)

        score_text = f"tire {annotation.score:.2f}"
        depth_text = f"depth {annotation.depth:.2f}"
        (_, score_h), _ = cv2.getTextSize(score_text, font, font_scale, font_thickness)
        (depth_w, depth_h), _ = cv2.getTextSize(depth_text, font, font_scale, font_thickness)
        text_y = y1 + max(score_h, depth_h) + text_padding
        depth_x = max(x1 + text_padding, x2 - depth_w - text_padding)
        cv2.putText(image_bgr, score_text, (x1 + text_padding, text_y), font, font_scale, tire_color, font_thickness, cv2.LINE_AA)
        cv2.putText(image_bgr, depth_text, (depth_x, text_y), font, font_scale, tire_color, font_thickness, cv2.LINE_AA)

        for stud in annotation.studs:
            sx, sy, sw, sh = stud.box
            stud_color = STUD_COLORS[stud.label_id]
            cv2.rectangle(image_bgr, (sx - sw // 2, sy - sh // 2), (sx + sw // 2, sy + sh // 2), stud_color, 2)

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
