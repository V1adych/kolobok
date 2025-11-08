import time
import logging
from typing import List, Optional


from tire_vision.text.index.db import TireModelDatabase
from tire_vision.config import IndexConfig
from tire_vision.options import IndexOptions
from models import IndexResult


class IndexPipeline:
    def __init__(self, config: IndexConfig):
        self.config = config
        self.database = TireModelDatabase(config)
        self.logger = logging.getLogger("tire_index_pipeline")
        self.logger.info("TireIndexPipeline initialized")

    def __call__(self, queries: List[str], options: Optional[IndexOptions] = None) -> List[IndexResult]:
        self.logger.info(f"Getting best matches for queries: {queries}")
        start_time = time.perf_counter()
        result = self.database.query(queries, options=options)
        result = result.to_dicts()
        result = [
            IndexResult(
                model_id=r["model_id"],
                model_name=r["model_name"],
                candidate_model_name=r["candidate_model_name"],
                candidate_model_score=r["candidate_model_score"],
                brand_id=r["brand_id"],
                brand_name=r["brand_name"],
                candidate_brand_name=r["candidate_brand_name"],
                candidate_brand_score=r["candidate_brand_score"],
                combined_score=r["combined_score"],
            )
            for r in result
        ]
        latency = time.perf_counter() - start_time
        self.logger.info(f"IndexPipeline completed in {latency:.4f} seconds")
        return result
