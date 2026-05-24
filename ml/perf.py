from functools import wraps
from datetime import datetime
import logging

from models import PerfStats


def _attach_stats(obj, start_time, end_time):
    stats = PerfStats(
        request_received_timestamp=start_time.isoformat(timespec="milliseconds"),
        request_completed_timestamp=end_time.isoformat(timespec="milliseconds"),
        total_time_seconds=(end_time - start_time).total_seconds(),
    )
    if hasattr(obj, "perf_stats"):
        obj.perf_stats = stats
        return

    raise ValueError("Object has no perf_stats attribute")


def get_perf_logger(logger: logging.Logger):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.info(f"{func.__name__}: starting")
            result = await func(*args, **kwargs)
            end_time = datetime.now()
            logger.info(f"{func.__name__}: completed in {end_time - start_time}")
            _attach_stats(result, start_time, end_time)
            return result

        return wrapper

    return decorator
