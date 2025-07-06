from itertools import product
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager, contextmanager
import logging

from sqlalchemy import (
    Table,
    Column,
    Integer,
    TEXT,
    MetaData,
    select,
    desc,
    and_,
    create_engine,
    text,
    func,
    union_all,
    literal,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


from tire_vision.config import IndexConfig

metadata = MetaData()


class TireModelDatabase:
    def __init__(self, config: IndexConfig):
        self.config = config
        self.logger = logging.getLogger("tire_db")
        self._models_table = None

        async_connection_url = (
            f"mysql+aiomysql://{config.db_user}:{config.db_password}@"
            f"{config.db_host}:{config.db_port}/{config.db_name}"
        )
        self.async_engine = create_async_engine(
            async_connection_url, pool_pre_ping=True, pool_recycle=3600, echo=False
        )
        self.async_session = sessionmaker(
            self.async_engine, class_=AsyncSession, expire_on_commit=False
        )

        sync_connection_url = (
            f"mysql+pymysql://{config.db_user}:{config.db_password}@"
            f"{config.db_host}:{config.db_port}/{config.db_name}"
        )
        self.sync_engine = create_engine(
            sync_connection_url, pool_pre_ping=True, pool_recycle=3600, echo=False
        )
        self.sync_session = sessionmaker(self.sync_engine, expire_on_commit=False)

    @property
    def models_table(self) -> Table:
        if self._models_table is None:
            self._models_table = Table(
                self.config.table_name,
                metadata,
                Column("id", Integer, primary_key=True),
                Column("name", TEXT, nullable=False),
                Column("parent_id", Integer, nullable=False, default=0),
            )
        return self._models_table

    @asynccontextmanager
    async def async_get_session(self):
        async with self.async_session() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    @contextmanager
    def get_session(self):
        with self.sync_session() as session:
            try:
                yield session
            except Exception:
                session.rollback()
                raise

    async def async_get_brands(
        self, query: str, limit: int = 10, confidence_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        async with self.async_get_session() as session:
            similarity_score = func.SIMILARITY_SCORE(
                self.models_table.c.name, query
            ).label("similarity_score")

            subquery = (
                select(
                    self.models_table.c.id, self.models_table.c.name, similarity_score
                )
                .where(self.models_table.c.parent_id == 0)
                .alias("subquery")
            )

            stmt = (
                select(subquery)
                .where(subquery.c.similarity_score >= confidence_threshold)
                .order_by(desc(subquery.c.similarity_score))
                .limit(limit)
            )

            result = await session.execute(stmt)
            return [
                {
                    "id": row.id,
                    "name": row.name,
                    "similarity_score": float(row.similarity_score or 0.0),
                }
                for row in result.fetchall()
            ]

    def get_brands(
        self, query: str, limit: int = 10, confidence_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            similarity_score = func.SIMILARITY_SCORE(
                self.models_table.c.name, query
            ).label("similarity_score")

            subquery = (
                select(
                    self.models_table.c.id, self.models_table.c.name, similarity_score
                )
                .where(self.models_table.c.parent_id == 0)
                .alias("subquery")
            )

            stmt = (
                select(subquery)
                .where(subquery.c.similarity_score >= confidence_threshold)
                .order_by(desc(subquery.c.similarity_score))
                .limit(limit)
            )

            result = session.execute(stmt)
            return [
                {
                    "id": row.id,
                    "name": row.name,
                    "similarity_score": float(row.similarity_score or 0.0),
                }
                for row in result.fetchall()
            ]

    async def async_get_brands_for_multiple_queries(
        self, queries: List[str], limit: int = 10, confidence_threshold: float = 0.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not queries:
            return {}

        individual_selects = []
        for q in queries:
            similarity_score = func.SIMILARITY_SCORE(self.models_table.c.name, q).label(
                "similarity_score"
            )

            subq = (
                select(
                    self.models_table.c.id,
                    self.models_table.c.name,
                    similarity_score,
                )
                .where(self.models_table.c.parent_id == 0)
                .alias("subq")
            )

            stmt = (
                select(
                    subq.c.id,
                    subq.c.name,
                    subq.c.similarity_score,
                    literal(q).label("search_term"),
                )
                .where(subq.c.similarity_score >= confidence_threshold)
                .order_by(desc(subq.c.similarity_score))
                .limit(limit)
            )

            individual_selects.append(stmt)

        final_stmt = union_all(*individual_selects)

        async with self.async_get_session() as session:
            result = await session.execute(final_stmt)
            output = {q: [] for q in queries}
            for row in result.mappings():
                output[row.search_term].append(
                    {
                        "id": row.id,
                        "name": row.name,
                        "similarity_score": float(row.similarity_score or 0.0),
                    }
                )
            return output

    def get_brands_for_multiple_queries(
        self, queries: List[str], limit: int = 10, confidence_threshold: float = 0.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not queries:
            return {}

        individual_selects = []
        for q in queries:
            similarity_score = func.SIMILARITY_SCORE(self.models_table.c.name, q).label(
                "similarity_score"
            )

            subq = (
                select(
                    self.models_table.c.id,
                    self.models_table.c.name,
                    similarity_score,
                )
                .where(self.models_table.c.parent_id == 0)
                .alias("subq")
            )

            stmt = (
                select(
                    subq.c.id,
                    subq.c.name,
                    subq.c.similarity_score,
                    literal(q).label("search_term"),
                )
                .where(subq.c.similarity_score >= confidence_threshold)
                .order_by(desc(subq.c.similarity_score))
                .limit(limit)
            )

            individual_selects.append(stmt)

        final_stmt = union_all(*individual_selects)

        with self.get_session() as session:
            result = session.execute(final_stmt)
            output = {q: [] for q in queries}
            for row in result.mappings():
                output[row.search_term].append(
                    {
                        "id": row.id,
                        "name": row.name,
                        "similarity_score": float(row.similarity_score or 0.0),
                    }
                )
            return output

    def _get_models_base_query(self, query: str):
        parent_table = self.models_table.alias("parent")
        similarity_score = func.SIMILARITY_SCORE(self.models_table.c.name, query).label(
            "similarity_score"
        )

        return select(
            self.models_table.c.id,
            self.models_table.c.name,
            self.models_table.c.parent_id.label("brand_id"),
            parent_table.c.name.label("brand_name"),
            similarity_score,
        ).select_from(
            self.models_table.join(
                parent_table, self.models_table.c.parent_id == parent_table.c.id
            )
        )

    async def async_get_models(
        self,
        query: str,
        brand_id: Optional[int] = None,
        limit: int = 10,
        confidence_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        async with self.async_get_session() as session:
            base_query = self._get_models_base_query(query)

            where_clauses = [self.models_table.c.parent_id != 0]
            if brand_id:
                where_clauses.append(self.models_table.c.parent_id == brand_id)

            subquery = base_query.where(and_(*where_clauses)).alias("subquery")

            stmt = (
                select(subquery)
                .where(subquery.c.similarity_score >= confidence_threshold)
                .order_by(desc(subquery.c.similarity_score))
                .limit(limit)
            )

            result = await session.execute(stmt)
            return [
                {
                    "id": row.id,
                    "name": row.name,
                    "brand_id": row.brand_id,
                    "brand_name": row.brand_name,
                    "similarity_score": float(row.similarity_score or 0.0),
                }
                for row in result.fetchall()
            ]

    def get_models(
        self,
        query: str,
        brand_id: Optional[int] = None,
        limit: int = 10,
        confidence_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            base_query = self._get_models_base_query(query)

            where_clauses = [self.models_table.c.parent_id != 0]
            if brand_id:
                where_clauses.append(self.models_table.c.parent_id == brand_id)

            subquery = base_query.where(and_(*where_clauses)).alias("subquery")

            stmt = (
                select(subquery)
                .where(subquery.c.similarity_score >= confidence_threshold)
                .order_by(desc(subquery.c.similarity_score))
                .limit(limit)
            )

            result = session.execute(stmt)
            return [
                {
                    "id": row.id,
                    "name": row.name,
                    "brand_id": row.brand_id,
                    "brand_name": row.brand_name,
                    "similarity_score": float(row.similarity_score or 0.0),
                }
                for row in result.fetchall()
            ]

    async def async_get_models_for_multiple_queries(
        self,
        queries: List[str],
        brand_ids: Optional[List[int]] = None,
        limit: int = 10,
        confidence_threshold: float = 0.0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not queries:
            return {}

        if brand_ids is None:
            brand_ids = [None]

        individual_selects = []
        for query, brand_id in product(queries, brand_ids):
            base_select = self._get_models_base_query(query)

            where_clauses = [self.models_table.c.parent_id != 0]
            if brand_id:
                where_clauses.append(self.models_table.c.parent_id == brand_id)

            subq = base_select.where(and_(*where_clauses)).alias("subq")

            stmt = (
                select(
                    subq.c.id,
                    subq.c.name,
                    subq.c.brand_id,
                    subq.c.brand_name,
                    subq.c.similarity_score,
                    literal(query).label("search_term"),
                )
                .where(subq.c.similarity_score >= confidence_threshold)
                .order_by(desc(subq.c.similarity_score))
                .limit(limit)
            )

            individual_selects.append(stmt)

        final_stmt = union_all(*individual_selects)

        async with self.async_get_session() as session:
            result = await session.execute(final_stmt)
            output = {query: [] for query in queries}
            for row in result.mappings():
                output[row.search_term].append(
                    {k: v for k, v in row.items() if k != "search_term"}
                )

        return output

    def get_models_for_multiple_queries(
        self,
        queries: List[str],
        brand_ids: Optional[List[int]] = None,
        limit: int = 10,
        confidence_threshold: float = 0.0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not queries:
            return {}

        if brand_ids is None:
            brand_ids = [None]

        individual_selects = []
        for query, brand_id in product(queries, brand_ids):
            base_select = self._get_models_base_query(query)

            where_clauses = [self.models_table.c.parent_id != 0]
            if brand_id:
                where_clauses.append(self.models_table.c.parent_id == brand_id)

            subq = base_select.where(and_(*where_clauses)).alias("subq")

            stmt = (
                select(
                    subq.c.id,
                    subq.c.name,
                    subq.c.brand_id,
                    subq.c.brand_name,
                    subq.c.similarity_score,
                    literal(query).label("search_term"),
                )
                .where(subq.c.similarity_score >= confidence_threshold)
                .order_by(desc(subq.c.similarity_score))
                .limit(limit)
            )

            individual_selects.append(stmt)

        final_stmt = union_all(*individual_selects)

        with self.get_session() as session:
            result = session.execute(final_stmt)
            output = {query: [] for query in queries}

            for row in result.mappings():
                output[row.search_term].append(
                    {k: v for k, v in row.items() if k != "search_term"}
                )

        return output

    async def async_close(self):
        await self.async_engine.dispose()

    def close(self):
        self.sync_engine.dispose()

    async def async_health_check(self) -> bool:
        try:
            async with self.async_get_session() as session:
                result = await session.execute(
                    text("SELECT SIMILARITY_SCORE('test', 'test') as score")
                )
                return result.scalar() == 1.0
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False

    def health_check(self) -> bool:
        try:
            with self.get_session() as session:
                result = session.execute(
                    text("SELECT SIMILARITY_SCORE('test', 'test') as score")
                )
                return result.scalar() == 1.0
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False

    def execute_query(self, query: str):
        with self.get_session() as session:
            result = session.execute(text(query))
            return result.fetchall()

    async def async_execute_query(self, query: str):
        async with self.async_get_session() as session:
            result = await session.execute(text(query))
            return result.fetchall()
