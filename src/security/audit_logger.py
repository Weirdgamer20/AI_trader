from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from ..ledger.database import DatabaseManager

class ProvenanceMemoryLogger:
    """
    Searchable Persistent Memory & Audit Trail.
    Categorizes all facts, calculations, hypotheses, and approvals
    to guarantee 100% traceability and prevent synthetic metric contamination.
    """
    CATEGORIES = {"FACT", "CALC", "HYPOTHESIS", "CONFIG", "RESEARCH", "INCIDENT"}

    def __init__(self, db: DatabaseManager):
        self.db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self.db.session() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_memory (
                    memory_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def record_memory(
        self,
        category: str,
        source: str,
        title: str,
        content: Dict[str, Any],
    ) -> str:
        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category {category}. Must be one of {self.CATEGORIES}")

        mem_id = f"mem-{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.session() as conn:
            conn.execute(
                """
                INSERT INTO system_memory (memory_id, category, source, title, content_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (mem_id, category, source, title, json.dumps(content), now_iso),
            )
        return mem_id

    def query_memories(self, category: str | None = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self.db.session() as conn:
            cur = conn.cursor()
            if category:
                cur.execute(
                    "SELECT * FROM system_memory WHERE category = ? ORDER BY created_at DESC LIMIT ?",
                    (category, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM system_memory ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                r["content"] = json.loads(r["content_json"])
            return rows
