import time
from typing import Dict, Any, List
import logging

from tire_vision.text.index.db import TireModelDatabase
from tire_vision.config import IndexConfig


class IndexPipeline:
    def __init__(self, config: IndexConfig):
        self.config = config
        self.database = TireModelDatabase(config)
        self.logger = logging.getLogger("tire_index_pipeline")
        self.logger.info("TireIndexPipeline initialized")

    def __call__(self, queries: List[str]) -> List[Dict[str, Any]]:
        self.logger.info(f"Getting best matches for queries: {queries}")
        start_time = time.perf_counter()
        result = self.database.query(queries)
        result = result.to_dicts()
        latency = time.perf_counter() - start_time
        self.logger.info(f"IndexPipeline completed in {latency:.4f} seconds")
        return result
