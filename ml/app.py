import base64
import io
from typing import Optional

from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from PIL import Image
import numpy as np

from models import (
    ThreadImageRequest,
    AnnotationImageRequest,
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
)
from perf import get_perf_logger
from tire_vision.options import TireThreadPipelineOptions, TireAnnotationPipelineOptions

import logging


app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)

logger = logging.getLogger("app")


@app.post("/api/v1/analyze_thread", response_model=ThreadAnalysisResponse)
@get_perf_logger(logger, async_mode=False)
def analyze_thread(
    req: ThreadImageRequest,
    token: str = Depends(verify_token),
):
    validate_image(req.image)

    image = np.array(Image.open(io.BytesIO(base64.b64decode(req.image))))
    result = get_thread_stats(image, options=req.thread_options)
    if result["success"] == 0:
        raise HTTPException(
            status_code=400, detail="Tire not found on the image, or it is too small"
        )

    image_with_annotations = add_annotations(result["vis_image"], result["studs"])

    pil_image = Image.fromarray(image_with_annotations)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return ThreadAnalysisResponse(
        thread_depth=result["depth"],
        studs=result["studs"],
        image=img_str,
    )


@app.post("/api/v1/extract_information", response_model=ExtractInformationResponse)
@get_perf_logger(logger, async_mode=False)
def extract_information(
    req: AnnotationImageRequest,
    token: str = Depends(verify_token),
):
    validate_image(req.image)

    image = np.array(Image.open(io.BytesIO(base64.b64decode(req.image))))

    result = extract_tire_info(image, options=req.annotation_options)

    return ExtractInformationResponse(
        strings=result["strings"],
        tire_size=result["tire_size"],
        index_results=result["index_results"],
    )


@app.post("/api/v1/bin/analyze_thread", response_model=ThreadAnalysisResponse)
@get_perf_logger(logger, async_mode=True)
async def analyze_thread_bin(
    image: UploadFile = File(...),
    options: Optional[TireThreadPipelineOptions] = Depends(parse_thread_options),
    token: str = Depends(verify_token),
):
    contents = await image.read()
    validate_image_bytes(contents)

    image_np = np.array(Image.open(io.BytesIO(contents)))
    result = get_thread_stats(image_np, options=options)
    if result["success"] == 0:
        raise HTTPException(
            status_code=400, detail="Tire not found on the image, or it is too small"
        )

    image_with_annotations = add_annotations(result["vis_image"], result["studs"])
    logger.info("/api/v1/bin/analyze_thread: thread pipeline completed")

    pil_image = Image.fromarray(image_with_annotations)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return ThreadAnalysisResponse(
        thread_depth=result["depth"],
        studs=result["studs"],
        image=img_str,
    )


@app.post("/api/v1/bin/extract_information", response_model=ExtractInformationResponse)
@get_perf_logger(logger, async_mode=True)
async def extract_information_bin(
    image: UploadFile = File(...),
    options: Optional[TireAnnotationPipelineOptions] = Depends(
        parse_annotation_options
    ),
    token: str = Depends(verify_token),
):
    contents = await image.read()
    validate_image_bytes(contents)

    image_np = np.array(Image.open(io.BytesIO(contents)))

    result = extract_tire_info(image_np, options=options)

    return ExtractInformationResponse(
        strings=result["strings"],
        tire_size=result["tire_size"],
        index_results=result["index_results"],
    )
