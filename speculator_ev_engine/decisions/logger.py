"""Universal decision logger — domain-agnostic schema."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

DEFAULT_DB_PATH = Path.home() / ".speculator_ev_engine" / "decisions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision TEXT NOT NULL,
    p_estimate REAL NOT NULL,
    ev_estimate REAL NOT NULL,
    stake REAL NOT NULL,
    outcome REAL,
    timestamp TEXT NOT NULL,
    domain TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '{}',
    notes TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_decisions_domain ON decisions(domain);
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);
CREATE INDEX IF NOT EXISTS idx_decisions_tags ON decisions(tags);
"""


@dataclass(frozen=True)
class Decision:
    """A single logged decision. Domain-agnostic.

    Attributes:
        decision: Description of the decision (hand ID, bet description, trade ID).
        p_estimate: Estimated probability of positive outcome.
        ev_estimate: Estimated expected value.
        stake: Amount wagered / risked.
        outcome: Actual outcome (None if not yet resolved).
        timestamp: When the decision was made.
        domain: "poker", "sports", or "markets".
        tags: Arbitrary key-value tags for filtering.
        notes: Free-text notes.
    """
    decision: str
    p_estimate: float
    ev_estimate: float
    stake: float
    outcome: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    domain: str = "general"
    tags: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_estimate <= 1.0):
            raise ValueError(f"p_estimate must be in [0,1], got {self.p_estimate}")
        if not np_isfinite(self.ev_estimate):
            raise ValueError(f"ev_estimate must be finite, got {self.ev_estimate}")
        if self.stake < 0:
            raise ValueError(f"stake must be non-negative, got {self.stake}")
        if self.domain not in {"poker", "sports", "markets", "general"}:
            raise ValueError(
                f"domain must be poker/sports/markets/general, got {self.domain}"
            )


class DecisionLogger:
    """SQLite-backed decision logger with schema enforcement.

    All writes go through validate-and-insert. Reads support filtering
    by domain, tags, date range, and outcome status.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize logger, creating DB and schema if needed.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.speculator_ev_engine/decisions.db
        """
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def log(self, decision: Decision) -> int:
        """Log a decision, returning its row ID.

        Args:
            decision: Validated Decision object.

        Returns:
            Integer row ID of the inserted record.
        """
        row = (
            decision.decision,
            decision.p_estimate,
            decision.ev_estimate,
            decision.stake,
            decision.outcome,
            decision.timestamp,
            decision.domain,
            json.dumps(decision.tags),
            decision.notes,
        )
        cursor = self._conn.execute(
            """INSERT INTO decisions
               (decision, p_estimate, ev_estimate, stake, outcome,
                timestamp, domain, tags, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            row,
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def resolve(self, row_id: int, outcome: float) -> None:
        """Set the outcome for a previously logged decision.

        Args:
            row_id: Row ID returned by log().
            outcome: Actual outcome value.
        """
        self._conn.execute(
            "UPDATE decisions SET outcome = ? WHERE id = ?",
            (outcome, row_id),
        )
        self._conn.commit()

    def query(
        self,
        domain: str | None = None,
        tag_filter: dict[str, str] | None = None,
        resolved_only: bool = False,
        min_timestamp: str | None = None,
        max_timestamp: str | None = None,
        limit: int = 1000,
    ) -> list[Decision]:
        """Query decisions with optional filters.

        Args:
            domain: Filter by domain.
            tag_filter: Filter by tag key-value pairs.
            resolved_only: Only return decisions with outcomes set.
            min_timestamp: ISO timestamp lower bound.
            max_timestamp: ISO timestamp upper bound.
            limit: Max records to return.

        Returns:
            List of Decision objects.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if domain is not None:
            conditions.append("domain = ?")
            params.append(domain)
        if resolved_only:
            conditions.append("outcome IS NOT NULL")
        if min_timestamp is not None:
            conditions.append("timestamp >= ?")
            params.append(min_timestamp)
        if max_timestamp is not None:
            conditions.append("timestamp <= ?")
            params.append(max_timestamp)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM decisions WHERE {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        results: list[Decision] = []
        for row in rows:
            tags = json.loads(row["tags"]) if row["tags"] else {}
            # Tag filtering in Python since SQLite JSON queries are fragile
            if tag_filter is not None:
                if not all(tags.get(k) == v for k, v in tag_filter.items()):
                    continue
            results.append(Decision(
                decision=row["decision"],
                p_estimate=row["p_estimate"],
                ev_estimate=row["ev_estimate"],
                stake=row["stake"],
                outcome=row["outcome"],
                timestamp=row["timestamp"],
                domain=row["domain"],
                tags=tags,
                notes=row["notes"] if row["notes"] is not None else "",
            ))
        return results

    def count(self, domain: str | None = None, resolved_only: bool = False) -> int:
        """Count decisions matching filters."""
        conditions: list[str] = []
        params: list[Any] = []
        if domain is not None:
            conditions.append("domain = ?")
            params.append(domain)
        if resolved_only:
            conditions.append("outcome IS NOT NULL")
        where = " AND ".join(conditions) if conditions else "1=1"
        row = self._conn.execute(
            f"SELECT COUNT(*) as cnt FROM decisions WHERE {where}", params
        ).fetchone()
        return int(row["cnt"])

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> DecisionLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def np_isfinite(x: float) -> bool:
    """Check if a float is finite without importing numpy at module level."""
    import math
    return math.isfinite(x)
