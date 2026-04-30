from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings
from .utils import ensure_dir, sha256_text, utc_now_iso, write_json

try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover - dependency may be absent during dry bootstrap
    bigquery = None


@dataclass
class QueryLogEntry:
    name: str
    sql_hash: str
    row_count: int
    runtime_seconds: float
    destination: str
    executed_at_utc: str
    dry_run: bool


class BigQueryClientWrapper:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._entries: list[QueryLogEntry] = []
        self._client = None

        if not settings.dry_run and bigquery is None:
            raise RuntimeError(
                "google-cloud-bigquery is not installed. Install dependencies before non-dry execution."
            )

    @property
    def query_log_path(self) -> Path:
        return (
            self.settings.paths.manifests_dir
            / f"query_log_{self.settings.run_id}.json"
        )

    def _ensure_client(self) -> Any:
        if self._client is None:
            if bigquery is None:
                raise RuntimeError("BigQuery client is unavailable in this environment.")
            self._client = bigquery.Client(project=self.settings.google_cloud_project)
        return self._client

    def query_to_dataframe(
        self,
        *,
        name: str,
        sql: str,
        dry_run_columns: list[str] | None = None,
        destination: str = "memory",
    ) -> pd.DataFrame:
        sql_hash = sha256_text(sql)

        if self.settings.dry_run:
            dataframe = pd.DataFrame(columns=dry_run_columns or [])
            self._entries.append(
                QueryLogEntry(
                    name=name,
                    sql_hash=sql_hash,
                    row_count=0,
                    runtime_seconds=0.0,
                    destination=destination,
                    executed_at_utc=utc_now_iso(),
                    dry_run=True,
                )
            )
            return dataframe

        client = self._ensure_client()
        start = time.perf_counter()
        query_job = client.query(sql)
        dataframe = query_job.to_dataframe(create_bqstorage_client=False)
        runtime = time.perf_counter() - start

        self._entries.append(
            QueryLogEntry(
                name=name,
                sql_hash=sql_hash,
                row_count=int(len(dataframe.index)),
                runtime_seconds=round(runtime, 3),
                destination=destination,
                executed_at_utc=utc_now_iso(),
                dry_run=False,
            )
        )
        return dataframe

    def execute_query(
        self,
        *,
        name: str,
        sql: str,
        destination: str = "query_job",
    ) -> None:
        sql_hash = sha256_text(sql)

        if self.settings.dry_run:
            self._entries.append(
                QueryLogEntry(
                    name=name,
                    sql_hash=sql_hash,
                    row_count=0,
                    runtime_seconds=0.0,
                    destination=destination,
                    executed_at_utc=utc_now_iso(),
                    dry_run=True,
                )
            )
            return

        client = self._ensure_client()
        start = time.perf_counter()
        query_job = client.query(sql)
        query_job.result()
        runtime = time.perf_counter() - start

        self._entries.append(
            QueryLogEntry(
                name=name,
                sql_hash=sql_hash,
                row_count=int(query_job.num_dml_affected_rows or 0),
                runtime_seconds=round(runtime, 3),
                destination=destination,
                executed_at_utc=utc_now_iso(),
                dry_run=False,
            )
        )

    def extract_to_parquet(
        self,
        *,
        name: str,
        sql: str,
        destination: Path,
        dry_run_columns: list[str] | None = None,
    ) -> Path:
        ensure_dir(destination.parent)
        dataframe = self.query_to_dataframe(
            name=name,
            sql=sql,
            dry_run_columns=dry_run_columns,
            destination=str(destination),
        )
        dataframe.to_parquet(destination, index=False)
        return destination

    def flush_query_log(self) -> Path:
        payload = [asdict(entry) for entry in self._entries]
        return write_json(self.query_log_path, payload)
