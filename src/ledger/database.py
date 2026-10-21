from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Any

CURRENT_SCHEMA_VERSION = 2

class DatabaseManager:
    """
    Unified Relational Database Manager.
    Provides deterministic transactional SQLite with WAL mode, foreign keys,
    and automated schema migrations.
    """
    def __init__(self, db_path: str | Path | None = None, connection_url: str | None = None):
        self.connection_url = connection_url or os.environ.get("DATABASE_URL")
        if self.connection_url and self.connection_url.startswith(("postgres://", "postgresql://")):
            # Dialect guard: PostgreSQL mode requires dedicated dialect driver
            raise NotImplementedError(
                "PostgreSQL dialect driver is not enabled in this deployment profile. "
                "Use local SQLite transactional engine for deterministic Windows execution."
            )

        if db_path is None:
            default_dir = Path(__file__).resolve().parent.parent.parent / "data"
            default_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = default_dir / "ledger.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._run_migrations()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    @contextmanager
    def session(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_column(self, cur: sqlite3.Cursor, table: str, column: str, col_def: str) -> None:
        cur.execute(f"PRAGMA table_info({table});")
        cols = [r[1] for r in cur.fetchall()]
        if column not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def};")

    def _verify_schema(self, cur: sqlite3.Cursor) -> None:
        """Verifies that all required tables and columns exist in the database."""
        required_schema = {
            "accounts": ["account_id", "account_type", "currency", "starting_balance", "available_balance", "locked_balance", "target_balance", "affordable_loss_ceiling", "loss_baseline_inr", "cumulative_realized_loss", "tds_receivable_inr", "is_active"],
            "orders": ["order_id", "client_order_id", "account_id", "market_pair", "side", "order_type", "status", "price", "quantity", "filled_quantity", "reserved_funds_inr", "reserved_quantity", "planned_risk_inr", "stop_loss_price", "take_profit_price", "rejection_reason", "idempotency_key"],
            "fills": ["fill_id", "order_id", "exchange_trade_id", "market_pair", "side", "price", "quantity", "fee_inr", "gst_on_fee_inr", "tds_deducted_inr", "net_proceeds_inr", "timestamp", "is_paper"],
            "positions": ["position_id", "account_id", "market_pair", "quantity", "reserved_quantity", "average_entry_price", "cost_basis_inr", "current_price", "unrealized_pnl_inr", "realized_pnl_inr", "is_open"],
            "journal_entries": ["entry_id", "transaction_id", "account_id", "account_name", "debit_inr", "credit_inr", "reference_id", "description", "timestamp"],
            "ledger_entries": ["entry_id", "account_id", "entry_type", "amount_inr", "balance_after_inr", "reference_id", "description", "timestamp"],
            "tax_records": ["record_id", "account_id", "financial_year", "fill_id", "market_pair", "sale_proceeds_inr", "cost_basis_inr", "economic_gain_inr", "taxable_gain_vda_inr", "tax_liability_30pct_inr", "tds_withheld_1pct_inr", "timestamp"],
            "audit_events": ["event_id", "event_type", "severity", "message", "details_json", "timestamp"],
        }
        for table, cols in required_schema.items():
            cur.execute(f"PRAGMA table_info({table});")
            existing = {r[1] for r in cur.fetchall()}
            if not existing:
                raise RuntimeError(f"Database migration verification failed: missing table '{table}'")
            missing = [c for c in cols if c not in existing]
            if missing:
                raise RuntimeError(f"Database migration verification failed: table '{table}' missing columns {missing}")

    def _run_migrations(self) -> None:
        """Applies versioned schema migrations with legacy column backfilling and schema verification."""
        with self.session() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL,
                    description TEXT NOT NULL
                );
                """
            )
            cur.execute("SELECT MAX(version) FROM schema_migrations;")
            row = cur.fetchone()
            current_ver = row[0] or 0

            if current_ver < 1:
                # Migration 1: Core transactional schema
                cur.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS accounts (
                        account_id TEXT PRIMARY KEY,
                        account_type TEXT NOT NULL,
                        currency TEXT NOT NULL DEFAULT 'INR',
                        starting_balance REAL NOT NULL DEFAULT 500.0,
                        available_balance REAL NOT NULL DEFAULT 500.0,
                        locked_balance REAL NOT NULL DEFAULT 0.0,
                        target_balance REAL NOT NULL DEFAULT 100000.0,
                        affordable_loss_ceiling REAL NOT NULL DEFAULT 250.0,
                        loss_baseline_inr REAL NOT NULL DEFAULT 0.0,
                        cumulative_realized_loss REAL NOT NULL DEFAULT 0.0,
                        tds_receivable_inr REAL NOT NULL DEFAULT 0.0,
                        is_active INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS orders (
                        order_id TEXT PRIMARY KEY,
                        client_order_id TEXT UNIQUE NOT NULL,
                        account_id TEXT NOT NULL,
                        market_pair TEXT NOT NULL,
                        side TEXT NOT NULL,
                        order_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        price REAL,
                        quantity REAL NOT NULL,
                        filled_quantity REAL NOT NULL DEFAULT 0.0,
                        reserved_funds_inr REAL NOT NULL DEFAULT 0.0,
                        reserved_quantity REAL NOT NULL DEFAULT 0.0,
                        planned_risk_inr REAL NOT NULL DEFAULT 0.0,
                        stop_loss_price REAL,
                        take_profit_price REAL,
                        rejection_reason TEXT,
                        idempotency_key TEXT UNIQUE,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                    );

                    CREATE TABLE IF NOT EXISTS fills (
                        fill_id TEXT PRIMARY KEY,
                        order_id TEXT NOT NULL,
                        exchange_trade_id TEXT UNIQUE,
                        market_pair TEXT NOT NULL,
                        side TEXT NOT NULL,
                        price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        fee_inr REAL NOT NULL DEFAULT 0.0,
                        gst_on_fee_inr REAL NOT NULL DEFAULT 0.0,
                        tds_deducted_inr REAL NOT NULL DEFAULT 0.0,
                        net_proceeds_inr REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        is_paper INTEGER NOT NULL DEFAULT 1,
                        FOREIGN KEY (order_id) REFERENCES orders(order_id)
                    );

                    CREATE TABLE IF NOT EXISTS positions (
                        position_id TEXT PRIMARY KEY,
                        account_id TEXT NOT NULL,
                        market_pair TEXT NOT NULL,
                        quantity REAL NOT NULL DEFAULT 0.0,
                        reserved_quantity REAL NOT NULL DEFAULT 0.0,
                        average_entry_price REAL NOT NULL DEFAULT 0.0,
                        cost_basis_inr REAL NOT NULL DEFAULT 0.0,
                        current_price REAL NOT NULL DEFAULT 0.0,
                        unrealized_pnl_inr REAL NOT NULL DEFAULT 0.0,
                        realized_pnl_inr REAL NOT NULL DEFAULT 0.0,
                        is_open INTEGER NOT NULL DEFAULT 1,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                    );

                    CREATE TABLE IF NOT EXISTS journal_entries (
                        entry_id TEXT PRIMARY KEY,
                        transaction_id TEXT NOT NULL,
                        account_id TEXT NOT NULL,
                        account_name TEXT NOT NULL,
                        debit_inr REAL NOT NULL DEFAULT 0.0,
                        credit_inr REAL NOT NULL DEFAULT 0.0,
                        reference_id TEXT,
                        description TEXT,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                    );

                    CREATE TABLE IF NOT EXISTS ledger_entries (
                        entry_id TEXT PRIMARY KEY,
                        account_id TEXT NOT NULL,
                        entry_type TEXT NOT NULL,
                        amount_inr REAL NOT NULL,
                        balance_after_inr REAL NOT NULL,
                        reference_id TEXT,
                        description TEXT,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                    );

                    CREATE TABLE IF NOT EXISTS tax_records (
                        record_id TEXT PRIMARY KEY,
                        account_id TEXT NOT NULL,
                        financial_year TEXT NOT NULL,
                        fill_id TEXT UNIQUE NOT NULL,
                        market_pair TEXT NOT NULL,
                        sale_proceeds_inr REAL NOT NULL,
                        cost_basis_inr REAL NOT NULL,
                        economic_gain_inr REAL NOT NULL,
                        taxable_gain_vda_inr REAL NOT NULL,
                        tax_liability_30pct_inr REAL NOT NULL,
                        tds_withheld_1pct_inr REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (account_id) REFERENCES accounts(account_id),
                        FOREIGN KEY (fill_id) REFERENCES fills(fill_id)
                    );

                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details_json TEXT,
                        timestamp TEXT NOT NULL
                    );
                    """
                )
                # Upgrade check for legacy pre-existing databases
                self._ensure_column(cur, "accounts", "locked_balance", "REAL NOT NULL DEFAULT 0.0")
                self._ensure_column(cur, "accounts", "tds_receivable_inr", "REAL NOT NULL DEFAULT 0.0")
                self._ensure_column(cur, "accounts", "affordable_loss_ceiling", "REAL NOT NULL DEFAULT 250.0")
                self._ensure_column(cur, "accounts", "loss_baseline_inr", "REAL NOT NULL DEFAULT 0.0")
                self._ensure_column(cur, "accounts", "cumulative_realized_loss", "REAL NOT NULL DEFAULT 0.0")

                self._ensure_column(cur, "orders", "reserved_funds_inr", "REAL NOT NULL DEFAULT 0.0")
                self._ensure_column(cur, "orders", "reserved_quantity", "REAL NOT NULL DEFAULT 0.0")
                self._ensure_column(cur, "orders", "planned_risk_inr", "REAL NOT NULL DEFAULT 0.0")
                self._ensure_column(cur, "orders", "idempotency_key", "TEXT UNIQUE")

                self._ensure_column(cur, "positions", "reserved_quantity", "REAL NOT NULL DEFAULT 0.0")
                self._ensure_column(cur, "positions", "cost_basis_inr", "REAL NOT NULL DEFAULT 0.0")
                self._ensure_column(cur, "positions", "current_price", "REAL NOT NULL DEFAULT 0.0")
                self._ensure_column(cur, "positions", "unrealized_pnl_inr", "REAL NOT NULL DEFAULT 0.0")
                self._ensure_column(cur, "positions", "realized_pnl_inr", "REAL NOT NULL DEFAULT 0.0")

                # Verify full schema before committing version 1
                self._verify_schema(cur)

                cur.execute(
                    "INSERT INTO schema_migrations (version, applied_at, description) VALUES (1, datetime('now'), 'Initial balanced transactional ledger');"
                )

            if current_ver < 2:
                # Migration 2: Index optimizations
                cur.executescript(
                    """
                    CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
                    CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id);
                    CREATE INDEX IF NOT EXISTS idx_journal_tx ON journal_entries(transaction_id);
                    """
                )
                cur.execute(
                    "INSERT INTO schema_migrations (version, applied_at, description) VALUES (2, datetime('now'), 'Indexes for order and journal lookups');"
                )
