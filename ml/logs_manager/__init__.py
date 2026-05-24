from functools import wraps
from typing import Any, Callable, Dict
import asyncio
import base64
import io
import os

import numpy as np
from PIL import Image

from logs_manager.mgr import LogsMgr
from logs_manager.config import LogsConfig

config = LogsConfig()
mgr = LogsMgr(config)


def _schedule_upload(image: np.ndarray, options: Dict[str, Any]) -> None:
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, mgr.upload_log, image, options)


def log_endpoint(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if os.environ.get("DISABLE_LOGGING") is not None:
            return await func(*args, **kwargs)

        req = kwargs.get("req")
        if req is not None:
            data = {key: value for key, value in req.model_dump().items() if key != "image"}
            result = await func(*args, **kwargs)
            raw = base64.b64decode(req.image)
            image = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        else:
            upload = kwargs["image"]
            contents = await upload.read()
            await upload.seek(0)
            options = kwargs.get("options")
            data = options.model_dump() if options is not None else {}
            result = await func(*args, **kwargs)
            image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

        _schedule_upload(image, data)
        return result

    return wrapper
