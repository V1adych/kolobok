from functools import wraps
from typing import Callable, Dict, Any
import asyncio

import numpy as np

from .mgr import LogsMgr
from .config import LogsConfig

mgr = LogsMgr(LogsConfig())
loop = asyncio.get_event_loop()


def log_wrapper(func: Callable[[np.ndarray], Dict[str, Any]]):
    @wraps(func)
    def wrapper(image: np.ndarray):
        result = func(image)
        loop.run_in_executor(None, mgr.upload_log, image, result, func.__name__)
        return result

    return wrapper
