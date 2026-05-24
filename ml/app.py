import base64
import io
from typing import Optional

from fastapi import FastAPI, Depends, File, UploadFile
from PIL import Image
import numpy as np

from models import (
    ThreadImageRequest,
    AnnotationImageRequest,
    TireResult,
    ThreadAnalysisResponse,
    ExtractInformationResponse,
)
from utils import (
    get_thread_stats,
    add_annotations,
    extract_tire_info,
    parse_thread_options,
    parse_annotation_options,
    verify_token,
    validate_image,
    validate_image_bytes,
    numpy_to_base64,
)
from perf import get_perf_logger
from logs_manager import log_endpoint
from tire_vision.options import TireThreadPipelineOptions, TireAnnotationPipelineOptions

import logging


app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)

logger = logging.getLogger("app")


def build_thread_response(result, image: np.ndarray) -> ThreadAnalysisResponse:
    image_with_annotations = add_annotations(image, result.tires)
    img_str = numpy_to_base64(image_with_annotations)
    tires = [
        TireResult(
            box=tire.box,
            score=tire.score,
            thread_depth=tire.depth,
            studs=tire.studs,
            num_studs=tire.num_studs,
            num_studs_classified=tire.num_studs_classified,
            fraction_healthy=tire.fraction_healthy,
        )
        for tire in result.tires
    ]
    return ThreadAnalysisResponse(tires=tires, image=img_str)


@app.post("/api/v1/analyze_thread", response_model=ThreadAnalysisResponse)
@get_perf_logger(logger)
@log_endpoint
async def analyze_thread(req: ThreadImageRequest, token: str = Depends(verify_token)):
    validate_image(req.image)

    image = np.array(Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB"))
    result = get_thread_stats(image, options=req.thread_options)
    return build_thread_response(result, image)


@app.post("/api/v1/extract_information", response_model=ExtractInformationResponse)
@get_perf_logger(logger)
@log_endpoint
async def extract_information(req: AnnotationImageRequest, token: str = Depends(verify_token)):
    validate_image(req.image)

    image = np.array(Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB"))

    result = extract_tire_info(image, options=req.annotation_options)

    return ExtractInformationResponse(strings=result.strings, tire_size=result.tire_size, index_results=result.index_results)


@app.post("/api/v1/bin/analyze_thread", response_model=ThreadAnalysisResponse)
@get_perf_logger(logger)
@log_endpoint
async def analyze_thread_bin(
    image: UploadFile = File(...),
    options: Optional[TireThreadPipelineOptions] = Depends(parse_thread_options),
    token: str = Depends(verify_token),
):
    contents = await image.read()
    validate_image_bytes(contents)

    image_np = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    result = get_thread_stats(image_np, options=options)
    logger.info("/api/v1/bin/analyze_thread: thread pipeline completed")
    return build_thread_response(result, image_np)


@app.post("/api/v1/bin/extract_information", response_model=ExtractInformationResponse)
@get_perf_logger(logger)
@log_endpoint
async def extract_information_bin(
    image: UploadFile = File(...),
    options: Optional[TireAnnotationPipelineOptions] = Depends(parse_annotation_options),
    token: str = Depends(verify_token),
):
    contents = await image.read()
    validate_image_bytes(contents)

    image_np = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    result = extract_tire_info(image_np, options=options)

    return ExtractInformationResponse(strings=result.strings, tire_size=result.tire_size, index_results=result.index_results)
