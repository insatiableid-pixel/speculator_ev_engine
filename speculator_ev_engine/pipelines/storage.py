"""SQLite-backed persistent storage for decisions, sessions, model outputs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


# TODO: Implement model output storage (predictions, features, metrics)
# TODO: Implement session metadata storage
# TODO: Implement query helpers for common analytical patterns


DEFAULT_STORAGE_PATH = Path.home() / ".speculator_ev_engine" / "storage.db"

STORAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    run_timestamp TEXT NOT NULL,
    parameters TEXT NOT NULL DEFAULT '{}',
    metrics TEXT NOT NULL DEFAULT '{}',
    predictions BLOB,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS session_meta (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    domain TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    summary TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);
"""


class Storage:
    """SQLite-backed storage for model outputs and session metadata.

    Decisions are stored via DecisionLogger; this handles the rest.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            db_path = DEFAULT_STORAGE_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.executescript(STORAGE_SCHEMA)

    def store_model_output(
        self,
        model_name: str,
        parameters: dict[str, Any],
        metrics: dict[str, float],
        run_timestamp: str | None = None,
    ) -> int:
        """Store a model run's outputs.

        Args:
            model_name: Model identifier.
            parameters: Model hyperparameters.
            metrics: Evaluation metrics.
            run_timestamp: When the run occurred.

        Returns:
            Row ID.
        """
        from datetime import datetime
        ts = run_timestamp or datetime.utcnow().isoformat()
        cursor = self._conn.execute(
            """INSERT INTO model_outputs
               (model_name, run_timestamp, parameters, metrics)
               VALUES (?, ?, ?, ?)""",
            (model_name, ts, json.dumps(parameters), json.dumps(metrics)),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_model_outputs(
        self, model_name: str | None = None, limit: int = 100
    ) -> pd.DataFrame:
        """Retrieve stored model outputs as a DataFrame.

        Args:
            model_name: Filter by model (None = all).
            limit: Maximum rows to return.

        Returns:
            DataFrame with model output records.
        """
        if model_name is not None:
            sql = "SELECT * FROM model_outputs WHERE model_name = ? ORDER BY run_timestamp DESC LIMIT ?"
            params = [model_name, limit]
        else:
            sql = "SELECT * FROM model_outputs ORDER BY run_timestamp DESC LIMIT ?"
            params = [limit]

        rows = self._conn.execute(sql, params).fetchall()
        columns = [desc[0] for desc in self._conn.execute(sql, params).description]
        return pd.DataFrame(rows, columns=columns)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Storage:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
