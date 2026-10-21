from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..ledger.database import DatabaseManager

class ExperimentRegistry:
    """
    Champion/Challenger Model Registry with automated rollback.
    Prevents unverified models from being promoted to production.
    """
    def __init__(self, db: DatabaseManager, registry_dir: Path | str | None = None):
        self.db = db
        self.registry_dir = Path(registry_dir or "models/registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.active_champion: Dict[str, Any] = {
            "model_id": "champ-baseline-hgb-v1",
            "algorithm": "hist_gradient_boosting",
            "horizon": 1,
            "val_accuracy": 0.5350,
            "val_mae": 1500.0,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        self.history: List[Dict[str, Any]] = [dict(self.active_champion)]

    def register_challenger(
        self,
        model_id: str,
        algorithm: str,
        horizon: int,
        val_accuracy: float,
        val_mae: float,
        hyperparameters: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Logs challenger performance and evaluates for gated promotion."""
        challenger = {
            "model_id": model_id,
            "algorithm": algorithm,
            "horizon": horizon,
            "val_accuracy": round(val_accuracy, 4),
            "val_mae": round(val_mae, 2),
            "hyperparameters": hyperparameters or {},
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }

        # Gated promotion check:
        # Challenger must have higher validation accuracy AND lower or equal MAE
        current_acc = self.active_champion["val_accuracy"]
        current_mae = self.active_champion["val_mae"]

        is_promoted = (val_accuracy >= current_acc + 0.005) and (val_mae <= current_mae)

        if is_promoted:
            challenger["promoted_at"] = datetime.now(timezone.utc).isoformat()
            self.history.append(dict(self.active_champion))
            self.active_champion = dict(challenger)

            # Record audit event
            with self.db.session() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO audit_events (event_id, event_type, severity, message, details_json, timestamp)
                    VALUES (?, 'MODEL_PROMOTION', 'INFO', ?, ?, ?)
                    """,
                    (
                        f"prom-{model_id}",
                        f"Challenger {model_id} ({algorithm}) promoted to Champion",
                        json.dumps(challenger),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

        return {
            "challenger_id": model_id,
            "promoted_to_champion": is_promoted,
            "active_champion_id": self.active_champion["model_id"],
            "comparison": {
                "challenger_acc": val_accuracy,
                "champion_acc": current_acc,
                "challenger_mae": val_mae,
                "champion_mae": current_mae,
            },
        }

    def rollback_to_previous_champion(self) -> Dict[str, Any]:
        """Rolls back to the immediately preceding approved champion."""
        if not self.history:
            return {"success": False, "message": "No previous champion available in history."}

        previous = self.history.pop()
        old_champion_id = self.active_champion["model_id"]
        self.active_champion = dict(previous)

        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.session() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO audit_events (event_id, event_type, severity, message, details_json, timestamp)
                VALUES (?, 'MODEL_PROMOTION', 'WARNING', ?, ?, ?)
                """,
                (
                    f"rollback-{int(datetime.now().timestamp())}",
                    f"Model rolled back from {old_champion_id} to {self.active_champion['model_id']}",
                    json.dumps(self.active_champion),
                    now_iso,
                ),
            )

        return {
            "success": True,
            "rolled_back_from": old_champion_id,
            "restored_champion": self.active_champion["model_id"],
        }
