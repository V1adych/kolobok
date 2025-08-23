import os
import base64
import io
from datetime import datetime
from functools import wraps

from fastapi import FastAPI, HTTPException, Depends, Security, status, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import numpy as np

from utils import get_thread_stats, add_annotations, extract_tire_info

import logging


app = FastAPI()
bearer_scheme = HTTPBearer()

API_TOKEN = os.environ["API_TOKEN"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)

logger = logging.getLogger("app")


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


class ImageRequest(BaseModel):
    image: str


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


def perf_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger.info(f"{func.__name__}: starting")
        result = func(*args, **kwargs)
        end_time = datetime.now()
        logger.info(f"{func.__name__}: completed in {end_time - start_time}")
        result["perf_stats"] = {
            "received_request": start_time.isoformat(),
            "completed_request": end_time.isoformat(),
            "duration": (end_time - start_time).total_seconds(),
        }
        return result
    return wrapper

def async_perf_logger(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger.info(f"{func.__name__}: starting")
        result = await func(*args, **kwargs)
        end_time = datetime.now()
        logger.info(f"{func.__name__}: completed in {end_time - start_time}")
        result["perf_stats"] = {
            "received_request": start_time.isoformat(),
            "completed_request": end_time.isoformat(),
            "duration": (end_time - start_time).total_seconds(),
        }
        return result
    return wrapper


@app.post("/api/v1/analyze_thread")
@perf_logger
def analyze_thread(
    req: ImageRequest,
    token: str = Depends(verify_token),
):
    validate_image(req.image)

    image = np.array(Image.open(io.BytesIO(base64.b64decode(req.image))))
    result = get_thread_stats(image)
    if result["success"] == 0:
        return result

    image_with_annotations = add_annotations(result["cropped_image"], result["spikes"])

    pil_image = Image.fromarray(image_with_annotations)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "success": 1,
        "thread_depth": result["depth"],
        "spikes": result["spikes"],
        "image": img_str,
    }


@app.post("/api/v1/extract_information")
@perf_logger
def extract_information(
    req: ImageRequest,
    token: str = Depends(verify_token),
):
    validate_image(req.image)

    image = np.array(Image.open(io.BytesIO(base64.b64decode(req.image))))

    result = extract_tire_info(image)

    return result


@app.post("/api/v1/bin/analyze_thread")
@async_perf_logger
async def analyze_thread_bin(
    image: UploadFile = File(...),
    token: str = Depends(verify_token),
):
    contents = await image.read()
    validate_image_bytes(contents)

    image_np = np.array(Image.open(io.BytesIO(contents)))
    result = get_thread_stats(image_np)
    if result["success"] == 0:
        return result

    image_with_annotations = add_annotations(result["cropped_image"], result["spikes"])
    logger.info("/api/v1/bin/analyze_thread: thread pipeline completed")

    pil_image = Image.fromarray(image_with_annotations)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "success": 1,
        "thread_depth": result["depth"],
        "spikes": result["spikes"],
        "image": img_str,
    }


@app.post("/api/v1/bin/extract_information")
@async_perf_logger
async def extract_information_bin(
    image: UploadFile = File(...),
    token: str = Depends(verify_token),
):
    contents = await image.read()
    validate_image_bytes(contents)

    image_np = np.array(Image.open(io.BytesIO(contents)))

    result = extract_tire_info(image_np)

    return result
