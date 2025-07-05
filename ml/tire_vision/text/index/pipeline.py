import time
from typing import Dict, Any, List
from collections import defaultdict
import logging

from tire_vision.text.index.db import TireModelDatabase
from tire_vision.config import TireIndexConfig


class TireIndexPipeline:
    def __init__(self, config: TireIndexConfig):
        self.config = config
        self.database = TireModelDatabase(config)
        self.logger = logging.getLogger("tire_index_pipeline")
        self.logger.info("TireIndexPipeline initialized")

    def run(self, queries: List[str]) -> Dict[str, Any]:
        self.logger.info(f"Running sync pipeline for queries: {queries}")
        start_time = time.time()
        brands_result = self.database.get_brands_for_multiple_queries(
            queries=queries,
            limit=self.config.max_query_results,
            confidence_threshold=self.config.similarity_threshold,
        )

        num_results = sum(len(results) for results in brands_result.values())
        self.logger.info(f"Number of result for brand search: {num_results}")

        brands = defaultdict(
            lambda: {
                "brand_id": None,
                "brand_name": None,
                "score": 0,
            }
        )

        for results in brands_result.values():
            for result in results:
                brands[result["id"]] = {
                    "brand_id": result["id"],
                    "brand_name": result["name"],
                    "score": max(
                        brands[result["id"]]["score"], result["similarity_score"]
                    ),
                }

        self.logger.info(f"Number of unique brands found: {len(brands)}")

        models = defaultdict(
            lambda: {
                "model_id": None,
                "model_name": None,
                "brand_id": None,
                "brand_name": None,
                "score": 0,
            }
        )

        models_result = self.database.get_models_for_multiple_queries(
            queries=queries,
            brand_ids=list(brands.keys()),
            limit=self.config.max_query_results,
            confidence_threshold=self.config.similarity_threshold,
        )

        num_results = sum(len(results) for results in models_result.values())
        self.logger.info(f"Number of result for model search: {num_results}")

        for results in models_result.values():
            for result in results:
                models[result["id"]] = {
                    "model_id": result["id"],
                    "model_name": result["name"],
                    "brand_id": result["brand_id"],
                    "brand_name": result["brand_name"],
                    "score": max(
                        models[result["id"]]["score"], result["similarity_score"]
                    )
                    * brands[result["brand_id"]]["score"],
                    "brand_id": result["brand_id"],
                }

        self.logger.info(f"Number of unique models found: {len(models)}")

        best_model = max(models.values(), key=lambda x: x["score"])

        end_time = time.time()
        self.logger.info(f"Time taken: {end_time - start_time} seconds")
        self.logger.info(f"Best model found: {best_model}")

        return best_model

    async def async_run(self, queries: List[str]) -> Dict[str, Any]:
        self.logger.info(f"Running async pipeline for queries: {queries}")
        start_time = time.time()
        brands_result = await self.database.async_get_brands_for_multiple_queries(
            queries=queries,
            limit=self.config.max_query_results,
            confidence_threshold=self.config.similarity_threshold,
        )

        num_results = sum(len(results) for results in brands_result.values())
        self.logger.info(f"Number of result for brand search: {num_results}")

        brands = defaultdict(
            lambda: {
                "brand_id": None,
                "brand_name": None,
                "score": 0,
            }
        )

        for results in brands_result.values():
            for result in results:
                brands[result["id"]] = {
                    "brand_id": result["id"],
                    "brand_name": result["name"],
                    "score": max(
                        brands[result["id"]]["score"], result["similarity_score"]
                    ),
                }

        num_results = sum(len(results) for results in models_result.values())
        self.logger.info(f"Number of result for model search: {num_results}")

        models = defaultdict(
            lambda: {
                "model_id": None,
                "model_name": None,
                "brand_id": None,
                "brand_name": None,
                "score": 0,
            }
        )

        models_result = await self.database.async_get_models_for_multiple_queries(
            queries=queries,
            brand_ids=list(brands.keys()),
            limit=self.config.max_query_results,
            confidence_threshold=self.config.similarity_threshold,
        )

        self.logger.info(f"Number of result for model search: {num_results}")

        for results in models_result.values():
            for result in results:
                models[result["id"]] = {
                    "model_id": result["id"],
                    "model_name": result["name"],
                    "brand_id": result["brand_id"],
                    "brand_name": result["brand_name"],
                    "score": max(
                        models[result["id"]]["score"], result["similarity_score"]
                    )
                    * brands[result["brand_id"]]["score"],
                }

        self.logger.info(f"Number of unique models found: {len(models)}")

        best_model = max(models.values(), key=lambda x: x["score"])
        end_time = time.time()
        self.logger.info(f"Time taken: {end_time - start_time} seconds")
        self.logger.info(f"Best model found: {best_model}")

        return best_model
