from functools import wraps
from typing import Callable, Dict, Any
import asyncio
import os

import numpy as np

from logs_manager.mgr import LogsMgr, serialize_exception
from logs_manager.config import LogsConfig

config = LogsConfig()
mgr = LogsMgr(config)
loop = asyncio.get_event_loop()


def log_wrapper(func: Callable[[np.ndarray], Dict[str, Any]]):
    @wraps(func)
    def wrapper(image: np.ndarray, *args, **kwargs):
        if os.environ.get("DISABLE_LOGGING", None) is not None:
            return func(image, *args, **kwargs)
        try:
            result = func(image, *args, **kwargs)
            loop.run_in_executor(None, mgr.upload_log, image, result, func.__name__)
            return result
        except Exception as e:
            error_log = serialize_exception(e)
            loop.run_in_executor(None, mgr.upload_log, image, error_log, f"{func.__name__}_error")
            raise e

    return wrapper
