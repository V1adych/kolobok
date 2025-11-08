from contextlib import contextmanager
import logging
import time
from pathlib import Path
from typing import Optional, List, Tuple, Literal
from dataclasses import replace

import numpy as np
from sqlalchemy import Table, Column, Integer, TEXT, MetaData, create_engine, text
from sqlalchemy.orm import sessionmaker

import polars as pl
import rapidfuzz


from tire_vision.config import IndexConfig
from tire_vision.options import IndexOptions


metadata = MetaData()


class TireModelDatabase:
    def __init__(self, config: IndexConfig):
        self.config = config
        self.logger = logging.getLogger("tire_db")
        self.models_table = Table(
            self.config.table_name,
            metadata,
            Column("id", Integer, primary_key=True),
            Column("name", TEXT, nullable=False),
            Column("parent_id", Integer, nullable=False, default=0),
        )

        connection_url = (
            f"mysql+pymysql://{config.db_user}:{config.db_password}@{config.db_host}:{config.db_port}/{config.db_name}"
        )
        self.engine = create_engine(connection_url, pool_pre_ping=True, pool_recycle=3600, echo=False)
        self.session = sessionmaker(self.engine, expire_on_commit=False)
        self.last_table_update = -float("inf")
        self.table_path = Path(self.config.table_cache_path)
        self._table = None

        self._similarity_metrics = {
            "levenshtein": rapidfuzz.distance.Levenshtein.normalized_similarity,
            "jaro_winkler": rapidfuzz.distance.JaroWinkler.similarity,
        }

        self._comb_metrics = {
            "product": self.product,
            "arithmetic_mean": self.arithmetic_mean,
            "harmonic_mean": self.harmonic_mean,
            "geometric_mean": self.geometric_mean,
            "euclidean": self.euclidean,
        }

        self.logger.info("TireModelDatabase initialized successfully")

    def calculate_score(self, name: str, queries: List[str]) -> List[Tuple[float, str]]:
        metric = self._similarity_metrics[self.config.options.similarity_metric]
        return list(map(lambda x: (metric(name, x[1]), x[0]), map(lambda x: (x, x.lower().strip()), queries)))

    def get_combined_score(
        self,
        model_score: pl.Series,
        brand_score: pl.Series,
        model_parent_id: pl.Series,
        brand_id: pl.Series,
    ) -> pl.Series:
        return (
            self._comb_metrics[self.config.options.comb_metric](model_score, brand_score)
            + pl.when(model_parent_id == brand_id).then(self.config.options.brand_model_match_bonus).otherwise(0)
        ) / (1 + self.config.options.brand_model_match_bonus)

    @contextmanager
    def get_session(self):
        with self.session() as session:
            yield session

    def execute_query(self, query: str):
        with self.get_session() as session:
            result = session.execute(text(query))
            return result.fetchall()

    def _load_table_from_db(self) -> Optional[pl.DataFrame]:
        try:
            table = pl.read_database(f"select * from {self.config.table_name}", self.engine)
            self.last_table_update = time.perf_counter()
            return table
        except Exception as e:
            self.logger.error(f"Error loading table from db: {e}")

        return None

    def _is_cache_expired(self) -> bool:
        return time.perf_counter() - self.last_table_update > self.config.table_cache_ttl_seconds

    def _save_table_to_disk(self):
        if self._table is not None:
            self.table_path.parent.mkdir(parents=True, exist_ok=True)
            self._table.write_parquet(self.table_path)
            self.logger.debug("Table saved to disk successfully")

    def _load_table_from_disk(self) -> Optional[pl.DataFrame]:
        if self.table_path.exists():
            table = pl.read_parquet(self.table_path)
            self.logger.info("Table loaded from disk cache")
            return table

        return None

    def _normalize_table_name(self, table: pl.DataFrame) -> pl.DataFrame:
        return table.with_columns(pl.col("name").str.to_lowercase().str.strip_chars(" ").alias("name_normalized"))

    @property
    def table(self) -> pl.DataFrame:
        if self._is_cache_expired():
            self.logger.info("Cache expired, attempting to update from database")
            fresh_table = self._load_table_from_db()
            if fresh_table is not None:
                self._table = self._normalize_table_name(fresh_table)
                self._save_table_to_disk()
                return self._table
            else:
                self.logger.warning("Failed to load from database, falling back to cache")

        if self._table is not None:
            return self._table

        self.logger.info("Table not loaded, attempting to load from disk")
        disk_table = self._load_table_from_disk()
        if disk_table is not None:
            self._table = self._normalize_table_name(disk_table)
            return self._table

        raise RuntimeError("Table unavailable: failed to load from database, disk cache, and RAM")

    def get_scores(self, queries: List[str]):
        df_queries = pl.DataFrame(
            {
                "query": queries,
                "query_normalized": list(map(lambda x: x.lower().strip(), queries)),
            }
        ).lazy()
        metric = self._similarity_metrics[self.config.options.similarity_metric]
        df = (
            self.table.lazy()
            .join(df_queries, how="cross")
            .with_columns(
                pl.struct([pl.col("name_normalized"), pl.col("query_normalized")])
                .map_elements(lambda x: metric(x["name_normalized"], x["query_normalized"]), return_dtype=pl.Float64)
                .alias("score"),
            )
        )
        return df

    def get_best_matches(self, df: pl.LazyFrame, kind: Literal["model", "brand"]):
        if kind == "brand":
            df = df.filter(pl.col("parent_id") == 0)
            limit_matches = self.config.options.max_brand_matches
        elif kind == "model":
            df = df.filter(pl.col("parent_id") != 0)
            limit_matches = self.config.options.max_model_matches
        else:
            raise ValueError(f"Invalid kind: {kind}")

        return (
            df.select(
                pl.col("id").alias(f"{kind}_id"),
                pl.col("name").alias(f"{kind}_name"),
                pl.col("parent_id"),
                pl.col("query_normalized").alias(f"candidate_{kind}_name"),
                pl.col("score").alias(f"candidate_{kind}_score"),
            )
            .sort(
                [
                    pl.col(f"candidate_{kind}_score"),
                    pl.col(f"{kind}_name").str.len_chars(),
                ],
                descending=[True, True],
            )
            .with_columns(
                pl.col(f"candidate_{kind}_score").rank(method="min", descending=True).over(f"{kind}_id").alias("rank")
            )
            .filter(pl.col("rank") <= self.config.options.max_distinct_matches)
            .drop("rank")
            .limit(limit_matches)
        )

    def query(self, queries: List[str], options: Optional[IndexOptions] = None):
        if options is not None:
            self.config = replace(self.config, options=options)
        suffix = "_right"
        df = self.get_scores(queries)
        df_model = self.get_best_matches(df, "model")
        df_brand = self.get_best_matches(df, "brand")

        col_templates = ["{kind}_id", "{kind}_name", "candidate_{kind}_name", "candidate_{kind}_score"]
        cols = [col.format(kind=kind) for kind in ["model", "brand"] for col in col_templates]
        cols.append("combined_score")

        df_model_brand = (
            df_model.join(df_brand, how="cross", suffix=suffix)
            .filter(pl.col("model_id") != pl.col("brand_id"))
            .with_columns(
                self.get_combined_score(
                    pl.col("candidate_model_score"),
                    pl.col("candidate_brand_score"),
                    pl.col("parent_id"),
                    pl.col("brand_id"),
                ).alias("combined_score"),
            )
            .select(*cols)
            .sort(
                [pl.col("combined_score"), pl.col("model_name").str.len_chars(), pl.col("brand_name").str.len_chars()],
                descending=[True, True, True],
            )
            .limit(self.config.options.max_query_results)
        )

        return df_model_brand.collect()

    @staticmethod
    def product(first, second):
        return first * second

    @staticmethod
    def arithmetic_mean(first, second):
        return (first + second) / 2

    @staticmethod
    def harmonic_mean(first, second):
        return 2 * first * second / (first + second + 1e-6)

    @staticmethod
    def geometric_mean(first, second):
        return np.sqrt(first * second)

    @staticmethod
    def euclidean(first, second):
        return 1 - np.sqrt((first - 1) ** 2 + (second - 1) ** 2)
